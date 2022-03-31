"""
    AbstractTrajectory

An abstract representation of a state and control trajectory. 

| **Required methods**  | **Brief description**  |
|:--------------------- |:---------------------- |
| `getstate(Z, t)` | Get the state at time `t`. |
| `getcontrol(Z, t)` | Get the control at time `t`. |
| `getinitialtime(Z)` | Get the initial time of the trajectory. |
| `getfinaltime(Z)` | Get the final time of the trajectory. |
"""
abstract type AbstractTrajectory end

getstate(Z::AbstractTrajectory, t) = throw(NotImplementedError("getstate not implemented for $(typeof(Z))."))
getcontrol(Z::AbstractTrajectory, t) = throw(NotImplementedError("getcontrol not implemented for $(typeof(Z))."))
getinitialtime(Z::AbstractTrajectory) = throw(NotImplementedError("getinitialtime not implemented for $(typeof(Z))."))
getfinaltime(Z::AbstractTrajectory) = throw(NotImplementedError("getfinaltime not implemented for $(typeof(Z))."))

"""
    SampledTrajectory{Nx,Nu,T,KP} <: AbstractTrajectory

A trajectory represented by a sample of knot points.
A vector of `AbstractKnotPoint`s of type `KP` with state dimension `Nx`,
control dimension `Nu`, and value type `T`.

Supports iteration and indexing.

# Constructors
    SampledTrajectory{Nx,Nu}(n, m; [equal, dt, tf, N])
    SampledTrajectory{Nx,Nu}(X, U; [dt, tf, N=length(X)])

    SampledTrajectory(n, m; [equal, dt, tf, N])
    SampledTrajectory(X, U; [dt, tf, N=length(X)])

where at least 2 of `dt`, `tf`, or `N` must be specified. The length `N` is automatically 
inferred when passing in state and control trajectories `X` and `U`, which can either be 
vectors of vectors or a 2D matrix whose 2nd dimension is time.
"""
struct SampledTrajectory{n,m,T,KP} <: AbstractTrajectory
    data::Vector{KP}
    times::Vector{T}
    function SampledTrajectory(Z::Vector{<:AbstractKnotPoint{n,m,<:Any,T}}) where {n,m,T}
        times = zeros(T, length(Z))
        for k = 1:length(Z)
            times[k] = time(Z[k])
        end
        new{n,m,T,eltype(Z)}(Z, times)
    end
end

#############################################
# AbstractTrajectory Interface 
#############################################
getstate(Z::SampledTrajectory, t) = state(Z.data[getk(Z, t)])
getcontrol(Z::SampledTrajectory, t) = control(Z.data[getk(Z, t)])
getinitialtime(Z::SampledTrajectory) = time(Z.data[1])
getfinaltime(Z::SampledTrajectory) = time(Z.data[end])

#############################################
# Iteration and Indexing 
#############################################
@inline Base.iterate(Z::SampledTrajectory) = iterate(Z.data)
@inline Base.iterate(Z::SampledTrajectory, k::Int) = iterate(Z.data, k)
Base.keys(Z::SampledTrajectory) = Base.keys(Z.data)
Base.IteratorSize(Z::SampledTrajectory) = Base.HasLength()
Base.IteratorEltype(Z::SampledTrajectory) = Base.IteratorEltype(Z.data)
@inline Base.eltype(Z::SampledTrajectory) = eltype(Z.data)
@inline Base.length(Z::SampledTrajectory) = length(Z.data)
@inline Base.size(Z::SampledTrajectory) = size(Z.data)
@inline Base.getindex(Z::SampledTrajectory, i) = Z.data[i]
@inline Base.setindex!(Z::SampledTrajectory, v, i) = Z.data[i] = v
@inline Base.firstindex(Z::SampledTrajectory) = 1
@inline Base.lastindex(Z::SampledTrajectory) = lastindex(Z.data)
Base.IndexStyle(::SampledTrajectory) = IndexLinear()

#############################################
# Constructors 
#############################################
function SampledTrajectory{Nx,Nu}(n::Int, m::Int; equal=false, kwargs...) where {Nx,Nu}
    times = _gettimeinfo(; kwargs...)
    dt = push!(diff(times), 0.0)
    Z = [KnotPoint{Nx,Nu}(n, m, fill(NaN, n + m), times[k], dt[k]) for k = 1:length(dt)]
    if equal
        Z[end].dt = Inf
    end
    return SampledTrajectory(Z)
end

function SampledTrajectory{Nx,Nu}(X::Matrix, U::Matrix; kwargs...) where {Nx,Nu}
    Xvec = [Vector(x) for x in eachcol(X)]
    Uvec = [Vector(u) for u in eachcol(U)]
    SampledTrajectory{Nx,Nu}(Xvec, Uvec; kwargs...)
end

function SampledTrajectory{Nx,Nu}(X::Vector, U::Vector; kwargs...) where {Nx, Nu}
    N = length(X)
    if !haskey(kwargs, :tf) && !haskey(kwargs, :dt) 
        error("Must specify either the time step or the total time.")
    end
    if haskey(kwargs, :N) && kwargs[:N] != N
        error("Specified an N inconsistent with the number of state vectors.")
    end
    times = _gettimeinfo(; N=N, kwargs...)
    dt = diff(times)
    Z = [KnotPoint{Nx,Nu}(length(X[k]), length(U[k]), [X[k]; U[k]], times[k], dt[k]) for k = 1:N-1]
    if length(U) == length(X)
        push!(Z, KnotPoint{Nx,Nu}(length(X[end]), length(U[end]), [X[end]; U[end]], times[N], Inf))
    else
        push!(Z, KnotPoint{Nx,Nu}(length(X[end]), length(U[end]), [X[end]; U[end]*0], times[N], 0.0))
    end
    return SampledTrajectory(Z)
end

function SampledTrajectory(X::Vector{<:StaticVector{Nx}}, U::Vector{<:StaticVector{Nu}}; 
              kwargs...) where {Nx,Nu}
    SampledTrajectory{Nx,Nu}(X, U; kwargs...)
end

SampledTrajectory(Z::SampledTrajectory) = Z

SampledTrajectory(args...; kwargs...) = SampledTrajectory{Any,Any}(args...; kwargs...)

#############################################
# Getters
#############################################
vectype(Z::SampledTrajectory{<:Any,<:Any,<:Any,KP}) where KP = vectype(KP)
has_terminal_control(Z::SampledTrajectory) = !RobotDynamics.is_terminal(Z[end])
state_dim(Z::SampledTrajectory{n}) where {n} = n
control_dim(Z::SampledTrajectory{<:Any,m}) where {m} = m
state_dim(Z::SampledTrajectory, k::Integer) = state_dim(Z[k]) 
control_dim(Z::SampledTrajectory, k::Integer) = control_dim(Z[k]) 
dims(Z::SampledTrajectory) = (state_dim.(Z), control_dim.(Z), length(Z))
getk(Z::SampledTrajectory, t::Real) = searchsortedfirst(Z.times, t)


"""
    num_vars(Z)

Total number of states and controls in a trajectory `Z`.
"""
function num_vars(Z::SampledTrajectory)
    mapreduce(+, Z) do z
        state_dim(z) + (is_terminal(z) ? 0 : control_dim(z))
    end
end

function num_vars(n::Int, m::Int, N::Int, equal::Bool=false)
    Nu = equal ? N : N-1
    return N*n + Nu*m
end

"""
    eachcontrol(Z)

Get the range of indices for valid controls.
"""
eachcontrol(Z::SampledTrajectory) = has_terminal_control(Z) ? Base.OneTo(length(Z)) : Base.OneTo(length(Z)-1)

"""
    states(Z)
    states(Z, i::Integer)
    states(Z, inds)

Get a list of all the state vectors for the trajectory `Z`. Passing 
an integer extracts a vector of the `i`th state. Passing a vector of 
integers provides a list of `N`-dimensional vectors, containing the 
time history for each state index in the vector.
"""
@inline states(Z::SampledTrajectory) = state.(Z)
states(Z::SampledTrajectory, inds::AbstractVector{<:Integer}) = [states(Z, i) for i in inds]
states(Z::SampledTrajectory, ind::Integer) = [state(z)[ind] for z in Z] 

"""
    controls(Z)
    controls(Z, i::Integer)
    controls(Z, inds)

Get a list of all the control vectors for the trajectory `Z`. Passing 
an integer extracts a vector of the `i`th control. Passing a vector of 
integers provides a list of `N`-dimensional vectors, containing the 
time history for each control index in the vector.
"""
function controls(Z::SampledTrajectory)
    return [control(Z[k]) for k in eachcontrol(Z) ]
end
controls(Z::SampledTrajectory, inds::AbstractVector{<:Integer}) = [controls(Z, i) for i in inds]
controls(Z::SampledTrajectory, ind::Integer) = [control(Z[k])[ind] for k in eachcontrol(Z)] 

"""
    gettimes(Z)

Get a vector of times for the entire trajectory.
"""
function gettimes(Z::SampledTrajectory)
    [time(z) for z in Z]
end

"""
    getdata(Z)

Get a list of the concatenated state and control vectors.
"""
getdata(Z::SampledTrajectory) = get_data.(Z)

#############################################
# Setters
#############################################
"""
    setstates!(Z, X)

Set the states of a trajectory `Z`, where `X` can be a vector of vectors or a matrix
of size `(n,N)`.
"""
function setstates!(Z::SampledTrajectory, X)
    for k in eachindex(Z)
        setstate!(Z[k], X[k])
    end
end

function setstates!(Z::SampledTrajectory, X::AbstractMatrix)
    for k in eachindex(Z)
        setstate!(Z[k], view(X, :, k))
    end
end

"""
    setcontrols!(Z, U)

Set the controls of a trajectory `Z`, where `U` can be a vector of vectors or a matrix
of size `(m,N)` or a single vector of size `(m,)`, which will be copied to all the 
time steps.
"""
function setcontrols!(Z::SampledTrajectory, U)
    for k in 1:length(Z)-1
        setcontrol!(Z[k], U[k])
    end
end

function setcontrols!(Z::SampledTrajectory, U::AbstractMatrix)
    for k in 1:length(Z)-1
        setcontrol!(Z[k], view(U, :, k))
    end
end

function setcontrols!(Z::SampledTrajectory, u::AbstractVector{<:Real})
    for k in 1:length(Z)-1
        setcontrol!(Z[k], u)
    end
end

"""
    setdata!(Z, V)

Set the concatenated state and control vector for each knot point in the trajectory `Z`. 
`V` may be either a vector of vectors or a 2D matrix.
"""
function setdata!(Z::SampledTrajectory, V)
    for k in 1:length(Z)-1
        setdata!(Z[k], V[k])
    end
end

function setdata!(Z::SampledTrajectory, V::AbstractMatrix)
    for k in 1:length(Z)-1
        setdata!(Z[k], view(V, :, k))
    end
end

"""
    settimes!(Z, ts)

Set the times for the entire trajectory. The time steps are automatically updated.
"""
function settimes!(Z::SampledTrajectory, ts)
    for k in eachindex(ts)
        Z[k].t = ts[k]
        k < length(ts) && (Z[k].dt = ts[k+1] - ts[k])
    end
end

"""
    set_dt!(Z, dt)

Set a constant time step for the entire trajectory.
"""
function set_dt!(Z::SampledTrajectory, dt::Real)
    t = Z[1].t
    for z in Z
        z.t = t
        if !is_terminal(z)
            z.dt = dt
            t += dt
        end
    end
    return t 
end

"""
    setinitialtime!(Z, t0)

Set the initial time of the trajectory, shifting all of the times by the required amount.
"""
function setinitialtime!(Z::SampledTrajectory, t0)
    t0_prev = time(Z[1])
    Δt = t0 - t0_prev
    for z in Z
        t = time(z)
        settime!(z, t + Δt)
    end
    return Z
end

#############################################
# Copying and Comparison 
#############################################
function Base.copy(Z::SampledTrajectory{Nx,Nu}) where {Nx,Nu}
    SampledTrajectory([KnotPoint{Nx,Nu}(z.n, z.m, copy(z.z), z.t, z.dt) for z in Z])
end

function Base.isapprox(Z1::SampledTrajectory, Z2::SampledTrajectory)
    all(zs->zs[1] ≈ zs[2], zip(Z1,Z2))
end

function Base.copyto!(Z::SampledTrajectory, Z0::SampledTrajectory)
    @assert length(Z) == length(Z0)
    V = vectype(Z)
    N = length(Z)
    for k = 1:N-1 
        if V <: StaticVector 
            Z[k].z = Z0[k].z
        else
            copyto!(Z[k].z, Z0[k].z)
        end
        Z[k].t = Z0[k].t
        Z[k].dt = Z0[k].dt
    end
    if is_terminal(Z[end])
        setstate!(Z[end], state(Z0[end]))
    else
        if V <: StaticVector 
            Z[N].z = Z0[N].z
        else
            copyto!(Z[N].z, Z0[N].z)
        end
    end
    Z[N].t = Z0[N].t
    Z[N].dt = Z0[N].dt
    Z
end

#############################################
# Misc
#############################################
function _gettimeinfo(;t0=0.0, tf=NaN, dt=NaN, N=0)
    if dt isa Vector
        return _gettimeinfo(dt, t0=t0, tf=tf, N=N)
    end
    Δt = zero(t0) 
    if isnan(tf) + isnan(dt) + (N == 0) > 1
        error("Must specify at least two of the following: dt, tf, N")
    end

    # Given tf, dt
    if (N==0) && !isnan(tf) && !isnan(dt)
        N = round(Int, tf / dt + 1)
        Δt = tf - t0
        dt = Δt / (N - 1)
    end

    # Given N, tf 
    if !isnan(tf) && isnan(dt)
        dt = tf / (N-1)
    end

    # Given N, dt
    if !isnan(dt) && isnan(tf)
        tf = dt * (N - 1)
    end
    Δt = tf - t0

    # Check consistency
    @assert tf == dt * (N-1) "Inconsistent time step and final time."
    @assert dt ≈ Δt / (N - 1) "Inconsistent time step and trajectory length."
    return range(t0, tf, length=N)
end

function _gettimeinfo(dt::AbstractVector; t0=0, tf=NaN, N=0)
    @assert all(x->x>zero(x), dt) "All time steps must be positive"
    if (N == 0)
        N = length(dt) + 1
    end
    if isnan(tf)
        tf = sum(dt) + t0
    end
    Δt = tf - t0
    @assert length(dt) == N-1 "Time step vector must have length of N-1, got $(length(dt)) with N=$N."
    @assert Δt ≈ sum(dt) "Elapsed time ($Δt) must be equal to the sum of the time steps (got $(sum(dt)))."
    return [t0; cumsum(dt)]
end

function shift_fill!(Z::SampledTrajectory, n=1)
    N = length(Z)
    isterm = is_terminal(Z[end])
    for k = 1+n:N 
        Z[k-n] = copy(Z[k])
    end
    xf = state(Z[N-n]) 
    uf = control(Z[N-n])
    dt = Z[N-n-1].dt
    for k = N-n:N
        setstate!(Z[k], xf) 
        Z[k].t = Z[k-1].t + dt
        if k == N && is_terminal(Z[k])
            Z[k].dt = 0
        else
            setcontrol!(Z[k], uf) 
            Z[k].dt = dt
        end
    end
end

#~~~~~~~~~~~~~~~~~~~~~~~~~~ FUNCTIONS ON TRAJECTORIES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# state_diff_jacobian!(model::AbstractModel, G, Z::AbstractTrajectory) = 
#     state_diff_jacobian!(statevectortype(model), model, G, Z)
# state_diff_jacobian!(::EuclideanState, model::AbstractModel, G, Z::AbstractTrajectory) = nothing

# function state_diff_jacobian!(::RotationState, model::AbstractModel, G, Z::AbstractTrajectory)
#     for k in eachindex(Z)
#         G[k] .= 0
#         state_diff_jacobian!(RotationState(), model, G[k], Z[k])
#     end
# end

function rollout!(sig::FunctionSignature, model::DiscreteDynamics, Z::AbstractTrajectory, x0=state(Z[1]))
    setstate!(Z[1], x0)
    for k = 2:length(Z)
        RobotDynamics.propagate_dynamics!(sig, model, Z[k], Z[k-1])
    end
end
