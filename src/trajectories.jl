abstract type AbstractTrajectory{n,m,T} <: AbstractVector{T} end

has_terminal_control(Z::AbstractTrajectory) = !RobotDynamics.is_terminal(Z[end])
traj_size(Z::AbstractTrajectory{n,m}) where {n,m} = n,m,length(Z)
num_vars(Z::AbstractTrajectory) = num_vars(traj_size(Z)..., has_terminal_control(Z))
eachcontrol(Z::AbstractTrajectory) = has_terminal_control(Z) ? Base.OneTo(length(Z)) : Base.OneTo(length(Z)-1)
state_dim(Z::AbstractTrajectory{n}) where {n} = n
control_dim(Z::AbstractTrajectory{<:Any,m}) where {m} = m
dims(Z::AbstractTrajectory) = traj_size(Z)

function num_vars(n::Int, m::Int, N::Int, equal::Bool=false)
    Nu = equal ? N : N-1
    return N*n + Nu*m
end

function Base.copyto!(dest::AbstractTrajectory, src::AbstractTrajectory)
	@assert traj_size(dest) == traj_size(src)
	for k = 1:length(src)
        copyto!(dest[k], src[k])
	end
	return src 
end

@inline states(Z::AbstractTrajectory) = state.(Z)
function controls(Z::AbstractTrajectory)
	return [control(Z[k]) for k in eachcontrol(Z) ]
end

states(Z::AbstractTrajectory, inds::AbstractVector{<:Integer}) = [states(Z, i) for i in inds]
controls(Z::AbstractTrajectory, inds::AbstractVector{<:Integer}) = [controls(Z, i) for i in inds]

states(Z::AbstractTrajectory, ind::Integer) = [state(z)[ind] for z in Z] 
controls(Z::AbstractTrajectory, ind::Integer) = [control(z)[ind] for z in Z] 

get_data(Z::AbstractTrajectory) = get_data.(Z)

function Base.isapprox(Z1::AbstractTrajectory, Z2::AbstractTrajectory)
    all(zs->zs[1] ≈ zs[2], zip(Z1,Z2))
end

function set_dt!(Z::AbstractTrajectory, dt::Real)
    t = Z[1].t
    for z in Z
        z.dt = dt
        z.t = t
        t += dt
    end
    return t 
end

"""
    Traj{n,m,T,KP}

A vector of `AbstractKnotPoint`s of type `KP` with state dimension `n`,
control dimension `m`, and value type `T`

Supports iteration and indexing.

# Constructors
    Traj(n, m, dt, N, equal=false)
    Traj(x, u, dt, N, equal=false)
    Traj(X, U, dt, t)
    Traj(X, U, dt)
"""
struct Traj{n,m,T,KP} <: AbstractTrajectory{n,m,T}
	data::Vector{KP}
	function Traj(Z::Vector{<:AbstractKnotPoint{n,m,<:Any,T}}) where {n,m,T}
		new{n,m,T,eltype(Z)}(Z)
	end
end

# AbstractArray interface
@inline Base.iterate(Z::Traj, k::Int) = iterate(Z.data, k)
Base.IteratorSize(Z::Traj) = Base.HasLength()
Base.IteratorEltype(Z::Traj) = Base.IteratorEltype(Z.data)
@inline Base.eltype(Z::Traj) = eltype(Z.data)
@inline Base.length(Z::Traj) = length(Z.data)
@inline Base.size(Z::Traj) = size(Z.data)
@inline Base.getindex(Z::Traj, i) = Z.data[i]
@inline Base.setindex!(Z::Traj, v, i) = Z.data[i] = v
@inline Base.firstindex(Z::Traj) = 1
@inline Base.lastindex(Z::Traj) = lastindex(Z.data)
Base.IndexStyle(::Traj) = IndexLinear()

Traj(Z::Traj) = Z

function Base.copy(Z::AbstractTrajectory{Nx,Nu}) where {Nx,Nu}
    Traj([KnotPoint{Nx,Nu}(copy(z.z), z.t, z.dt) for z in Z])
end

function Traj(n::Int, m::Int, dt::AbstractFloat, N::Int; equal=false)
    x = NaN*@SVector ones(n)
    u = @SVector zeros(m)
    Traj(x,u,dt,N; equal=equal)
end

function Traj(x::SVector, u::SVector, dt::AbstractFloat, N::Int; equal=false)
    equal ? uN = N : uN = N-1
    Z = [KnotPoint(x,u,(k-1)*dt,dt) for k = 1:uN]
    if !equal
        m = length(u)
        push!(Z, KnotPoint(x,u*0,(N-1)*dt,0.0))
    end
    return Traj(Z)
end

function Traj(X::Vector, U::Vector, dt::Vector, t=cumsum(dt) .- dt[1])
    n,m = length(X[1]), length(U[1])
    Z = [KnotPoint{n,m}(n,m,[X[k]; U[k]], t[k], dt[k]) for k = 1:length(U)]
    if length(U) == length(X)-1
        push!(Z, KnotPoint{n,m}(n,m,[X[end]; U[1]*0],t[end],0.0))
    end
    return Traj(Z)
end

function Traj(X::Matrix, U::Matrix; kwargs...)
    Xvec = [Vector(x) for x in eachcol(X)]
    Uvec = [Vector(u) for u in eachcol(U)]
    Traj(Xvec, Uvec; kwargs...)
end

function Traj(X::Vector, U::Vector; tf::Real=NaN, dt::Real=NaN)
    n,m = length(X[1]), length(U[1])
    N = length(X)
    if isnan(tf) && isnan(dt)
        error("Must specify either the time step or the total time.")
    end
    if !isnan(tf) && isnan(dt)
        dt = tf / (N-1)
    end
    if !isnan(dt) && isnan(tf)
        tf = dt * (N - 1)
    end
    if !isnan(dt) && !isnan(tf)
        @assert tf == dt * (N-1) "Inconsistent time step and final time."
    end
    Z = [KnotPoint{n,m}(n,m, [X[k]; U[k]], (k-1)*dt, dt) for k = 1:length(U)]
    if length(U) == length(X)-1
        push!(Z, KnotPoint{n,m}(n,m,[X[end]; U[1]*0], tf, 0.0))
    end
    return Traj(Z)
end

function setstates!(Z::Traj, X)
    for k in eachindex(Z)
		setstate!(Z[k], X[k])
    end
end

function setstates!(Z::Traj, X::AbstractMatrix)
    for k in eachindex(Z)
		setstate!(Z[k], X[:,k])
    end
end

function setcontrols!(Z::AbstractTrajectory, U)
    for k in 1:length(Z)-1
		setcontrol!(Z[k], U[k])
    end
end

function setcontrols!(Z::AbstractTrajectory, U::AbstractMatrix)
    for k in 1:length(Z)-1
		setcontrol!(Z[k], U[:,k])
    end
end

function setcontrols!(Z::AbstractTrajectory, u::SVector)
    for k in 1:length(Z)-1
		setcontrol!(Z[k], u)
    end
end

function settimes!(Z::AbstractTrajectory, ts)
    for k in eachindex(ts)
        Z[k].t = ts[k]
        k < length(ts) && (Z[k].dt = ts[k+1] - ts[k])
    end
end

function gettimes(Z::Traj)
    [z.t for z in Z]
end

function shift_fill!(Z::Traj, n=1)
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

function Base.copyto!(Z::Traj, Z0::Traj)
	@assert length(Z) == length(Z0)
    N = length(Z)
	for k = 1:N-1 
		copyto!(Z[k].z, Z0[k].z)
	end
    if is_terminal(Z[end])
        setstate!(Z[end], state(Z0[end]))
    else
		copyto!(Z[k].z, Z0[k].z)
    end
    Z
end

function Base.copyto!(Z::Traj{Nx,Nu,T,KP}, Z0::Traj{Nx,Nu,T,KP}) where {Nx,Nu,T,KP<:KnotPoint{Nx,Nu,<:StaticVector}}
	@assert length(Z) == length(Z0)
	for k in eachindex(Z)
		Z[k].z = Z0[k].z
	end
end

#~~~~~~~~~~~~~~~~~~~~~~~~~~ FUNCTIONS ON TRAJECTORIES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

state_diff_jacobian!(model::AbstractModel, G, Z::AbstractTrajectory) = 
    state_diff_jacobian!(statevectortype(model), model, G, Z)
state_diff_jacobian!(::EuclideanState, model::AbstractModel, G, Z::AbstractTrajectory) = nothing

function state_diff_jacobian!(::RotationState, model::AbstractModel, G, Z::AbstractTrajectory)
	for k in eachindex(Z)
		G[k] .= 0
		state_diff_jacobian!(RotationState(), model, G[k], Z[k])
	end
end

function rollout!(sig::FunctionSignature, model::DiscreteDynamics, Z::AbstractTrajectory, x0=state(Z[1]))
    setstate!(Z[1], x0)
    for k = 2:length(Z)
        RobotDynamics.propagate_dynamics!(sig, model, Z[k], Z[k-1])
    end
end
