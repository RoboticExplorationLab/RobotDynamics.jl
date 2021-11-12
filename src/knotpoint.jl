# struct StateControl{Nx,Nu,V,T} <: AbstractArray{T,1}
#     n::Int
#     m::Int
#     z::V
#     function (::Type{<:StateControl{Nx,Nu}})(z::V) where {Nx,Nu,V}
#         @assert Nx > 0
#         @assert Nu > 0
#         new{Nx,Nu,V,eltype(V)}(Nx,Nu,z)
#     end
#     function StateControl(n::Integer, m::Integer, z::V) where V
#         @assert n > 0
#         @assert m > 0
#         new{Any,Any,V,eltype(V)}(n,m,z)
#     end
#     function StateControl{Nx,Nu}(n::Integer, m::Integer, z::V) where {Nx,Nu,V}
#         new{Nx,Nu,V,eltype(V)}(n,m,z)
#     end
# end
# function StateControl(x, u)
#     return StateControl(length(x), length(u), [x; u])
# end
# function StateControl(x::StaticVector{Nx}, u::StaticVector{Nu}) where {Nx, Nu}
#     return StateControl{Nx,Nu}([x; u])
# end
# const StaticStateControl{n,m,nm,T} = StateControl{n,m,SVector{nm,T},T} where {n,m,nm,T}

# state(z::StateControl) = view(z.z, 1:z.n)
# control(z::StateControl) = view(z.z, z.n+1:z.n+z.m)
# state(z::StaticStateControl{Nx,Nu}) where {Nx,Nu} = z.z[SOneTo(Nx)] 
# control(z::StaticStateControl{Nx,Nu}) where {Nx,Nu} = z.z[SVector{Nu}(Nx+1:Nx+Nu)] 

# getstate(z::StateControl{Any,Any}, v) = view(v, 1:z.n)
# getcontrol(z::StateControl{Any,Any}, v) = view(v, z.n+1:z.n+z.m)
# getstate(z::StateControl{Nx,Nu}, v) where {Nx,Nu} = v[SOneTo(Nx)] 
# getcontrol(z::StateControl{Nx,Nu}, v) where {Nx,Nu} = v[SVector{Nu}(Nx+1:Nx+Nu)] 

# @inline getdata(z::StateControl) = z.z

# @generated function StaticArrays.SVector(z::StateControl{n,m}) where {n,m}
#     elements = [:(z.z[$i]) for i = 1:n+m]
#     :(SVector{$(n+m)}($(elements...)))
# end

# # Array interface
# Base.size(z::StateControl{n,m}) where {n,m} = (n+m,)
# Base.size(z::StateControl{Any,Any}) = (z.n + z.m,)
# Base.getindex(z::StateControl, i::Int) = z.z[i]
# Base.setindex!(z::StateControl, v, i::Integer) = z.z[i] = v
# Base.IndexStyle(z::StateControl) = IndexLinear()
# Base.similar(z::StateControl{Nx,Nu,V,T}, ::Type{S}, dims::Dims) where {Nx,Nu,V,T,S} = 
#     StateControl{Nx,Nu}(z.n, z.m, similar(z.z, S, dims))
# Base.similar(z::StaticStateControl{Nx,Nu,V,T}, ::Type{S}, dims::Dims) where {Nx,Nu,V,T,S} = 
#     StateControl{Nx,Nu}(z.n, z.m, similar(MVector{Nx+Nu,S}))

# mutable struct KnotPointData
#     n::Int       # number of states
#     m::Int       # number of control
#     t::Float64   # time
#     dt::Float64  # time step
# end
# state_dim(kp::KnotPointData) = kp.n
# control_dim(kp::KnotPointData) = kp.m
# time(kp::KnotPointData) = kp.t
# time_step(kp::KnotPointData) = kp.dt

abstract type AbstractKnotPoint{Nx,Nu,V,T} <: AbstractVector{T} end
# state_dim
# control_dim
# getparams
# getdata

dims(z::AbstractKnotPoint) = (state_dim(z), control_dim(z))
getstate(z::AbstractKnotPoint, v) = view(v, 1:state_dim(z))
getcontrol(z::AbstractKnotPoint, v) = begin n,m = dims(z); view(v, n+1:n+m) end
getstate(z::AbstractKnotPoint{Nx,Nu,<:StaticVector}, v) where {Nx,Nu} = v[SVector{Nx}(1:Nx)]
getcontrol(z::AbstractKnotPoint{Nx,Nu,<:StaticVector}, v) where {Nx,Nu} = v[SVector{Nu}(Nx+1:Nx+Nu)]

state(z::AbstractKnotPoint) = getstate(z, getdata(z))
control(z::AbstractKnotPoint) = getcontrol(z, getdata(z))

setdata!(z::AbstractKnotPoint, v) = z.z .= v
setdata!(z::AbstractKnotPoint{<:Any,<:Any,<:SVector}, v) = z.z = v

setstate!(z::AbstractKnotPoint, x) = state(z) .= x
setcontrol!(z::AbstractKnotPoint, u) = control(z) .= u
setstate!(z::AbstractKnotPoint{<:Any,<:Any,<:SVector}, x) = setdata!(z, [x; control(z)])
setcontrol!(z::AbstractKnotPoint{<:Any,<:Any,<:SVector}, u) = setdata!(z, [state(z); u])

time(z::AbstractKnotPoint) = getparams(z).t 
timestep(z::AbstractKnotPoint) = getparams(z).dt 

# Array interface
Base.size(z::AbstractKnotPoint) = (state_dim(z) + control_dim(z), )
Base.getindex(z::AbstractKnotPoint, i::Int) = Base.getindex(getdata(z), i) 
Base.setindex!(z::AbstractKnotPoint, v, i::Integer) = Base.setindex!(getdata(z), v, i) 
Base.IndexStyle(z::AbstractKnotPoint) = IndexLinear()

for (name,mutable) in [(:KnotPoint, true), (:StaticKnotPoint, false)]
    expr = quote
        struct $name{Nx,Nu,V,T} <: AbstractKnotPoint{Nx,Nu,V,T}
            z::V
            t::Float64
            dt::Float64
            n::Int
            m::Int
            function $name{Nx,Nu}(z::V,t,dt) where {Nx,Nu,V} 
                @assert Nx > 0
                @assert Nu > 0
                new{Nx,Nu,V,eltype(V)}(z, t, dt, Nx, Nu)
            end
            function $name(n::Integer, m::Integer, z::V, t, dt) where V
                @assert n > 0
                @assert m > 0
                new{Any,Any,V,eltype(V)}(z, t, dt, n, m)
            end
            function $name{Nx,Nu}(n::Integer, m::Integer, z::V, t, dt) where {Nx,Nu,V} 
                Nx != Any && @assert Nx == n
                Nu != Any && @assert Nu == m
                new{Nx,Nu,V,eltype(V)}(z, t, dt, n, m)
            end
        end
        function $name(x::StaticVector{Nx}, u::StaticVector{Nu}, t, dt) where {Nx, Nu}
            KnotPoint{Nx,Nu}([x;u], t, dt)
        end
        function $name(x::AbstractVector, u::AbstractVector, t, dt)
            KnotPoint(length(x), length(u), [x;u], t, dt)
        end

        state_dim(z::$name{Any}) = z.n
        control_dim(z::$name{<:Any,Any}) = z.m
        state_dim(z::$name{Nx}) where Nx = Nx 
        control_dim(z::$name{<:Any,Nu}) where Nu = Nu 

        @inline getparams(z::$name) = (t=z.t, dt=z.dt)
        @inline getdata(z::$name) = z.z

        # Array interface
        Base.similar(z::$name{Nx,Nu,V,T}, ::Type{S}, dims::Dims) where {Nx,Nu,V,T,S} = 
            KnotPoint{Nx,Nu}(z.n, z.m, similar(z.z, S, dims), z.t, z.dt)
        Base.similar(z::$name{Nx,Nu,<:SVector,T}, ::Type{S}, dims::Dims) where {Nx,Nu,T,S} = 
            KnotPoint{Nx,Nu}(z.n, z.m, similar(MVector{Nx+Nu,S}))
    end
    # Set struct mutability
    if mutable
        struct_expr = expr.args[2]
        @assert struct_expr.head == :struct
        struct_expr.args[1] = mutable
    end
    eval(expr)
end

function StaticKnotPoint(z::AbstractKnotPoint{Nx,Nu}, v::AbstractVector) where {Nx,Nu}
    StaticKnotPoint{Nx,Nu}(state_dim(z), control_dim(z), v, time(z), timestep(z))
end
@inline setdata(z::StaticKnotPoint, v) = StaticKnotPoint(z, v)
setstate(z::StaticKnotPoint, x) = setdata(z, [x; control(z)])
setcontrol(z::StaticKnotPoint, u) = setdata(z, [state(z); u)

# struct StaticKnotPoint{Nx,Nu,T,V} <: AbstractKnotPoint{Nx,Nu,T,V}
#     z::V
#     t::Float64
#     dt::Float64
#     n::Int
#     m::Int
#     function StaticKnotPoint{Nx,Nu}(z::V,t,dt) where {Nx,Nu,V} 
#         new{Nx,Nu,eltype(V),V}(z, t, dt, Nx, Nu)
#     end
#     function StaticKnotPoint(n::Integer, m::Integer, z::V, t, dt) where V
#         new{Any,Any,eltype(V),V}(z, t, dt, n, m)
#     end
# end
# function StaticKnotPoint(x::StaticVector{Nx}, u::StaticVector{Nu}, t, dt) where {Nx, Nu}
#     KnotPoint{Nx,Nu}([x;u], t, dt)
# end
# function StaticKnotPoint(x::AbstractVector, u::AbstractVector, t, dt)
#     KnotPoint(length(x), length(u), [x;u], t, dt)
# end

# state_dim(z::StaticKnotPoint{Any}) = z.n
# control_dim(z::StaticKnotPoint{<:Any,Any}) = z.m
# state_dim(z::StaticKnotPoint{Nx}) where Nx = Nx 
# control_dim(z::StaticKnotPoint{<:Any,Nu}) where Nu = Nu 

# getparams(z::StaticKnotPoint) = (t=z.t, dt=z.dt)
# getdata(z::StaticKnotPoint) = z.z

# export
#     KnotPoint,
#     state,
#     control

# """
# 	AbstractKnotPoint{T,n,m}

# Stores the states, controls, time, and time step at a single point along a trajectory of a
# forced dynamical system with `n` states and `m` controls.

# # Interface
# All instances of `AbstractKnotPoint` should support the following methods:

# 	state(z)::StaticVector{n}     # state vector
# 	control(z)::StaticVector{m}   # control vector
# 	z.t::Real                     # time
# 	z.dt::Real                    # time to next point (time step)

# By default, it is assumed that if `z.dt == 0` the point is the last point in the trajectory.

# Alternatively, the methods `state` and `control` will be automatically defined if the
# following fields are present:
# - `z.z`: the stacked vector `[x;u]`
# - `z._x`: the indices of the states, such that `x = z.z[z._x]`
# - `z._u`: the indices of the controls, such that `x = u.z[z._u]`
# """
# abstract type AbstractKnotPoint{T,N,M} end

# """
# 	state(::AbstractKnotPoint)

# Return the `n`-dimensional state vector
# """
# @inline state(z::AbstractKnotPoint) = z.z[z._x]

# """
# 	control(::AbstractKnotPoint)

# Return the `m`-dimensional control vector
# """
# @inline control(z::AbstractKnotPoint) = z.z[z._u]

# """
# 	is_terminal(::AbstractKnotPoint)::Bool

# Determine if the knot point is the terminal knot point, which is the case when `z.dt == 0`.
# """
# @inline is_terminal(z::AbstractKnotPoint) = z.dt == 0

# """
# 	get_z(::AbstractKnotPoint)

# Returns the stacked state-control vector `z`, or just the state vector if `is_terminal(z) == true`.
# """
# @inline get_z(z::AbstractKnotPoint) = is_terminal(z) ? state(z) : z.z

# """
# 	set_state!(z::AbstractKnotPoint, x::AbstractVector)

# Set the state in `z` to `x`.
# """
# set_state!(z::AbstractKnotPoint, x) = for i in z._x; z.z[i] = x[i]; end

# """
# 	set_control!(z::AbstractKnotPoint, u::AbstractVector)

# Set the controls in `z` to `u`.
# """
# set_control!(z::AbstractKnotPoint, u) = for (i,j) in enumerate(z._u); z.z[j] = u[i]; end

# """
# 	set_z!(z::AbstractKnotPoint, z_::AbstractVector)

# Set both the states and controls in `z` from the stacked state-control vector `z_`, unless
# `is_terminal(z)`, in which case `z_` is assumed to be the terminal states.
# """
# set_z!(z::AbstractKnotPoint, z_) = is_terminal(z) ? set_state!(z, z_) : copyto!(z.z, z_)

# function Base.isapprox(z1::AbstractKnotPoint, z2::AbstractKnotPoint)
#     get_z(z1) ≈ get_z(z2) && z1.t ≈ z2.t && z1.dt ≈ z2.dt
# end

# """
# 	GeneralKnotPoint{T,n,m,V} <: AbstractKnotPoint{T,n,m}

# A mutable instantiation of the `AbstractKnotPoint` interface where the joint vector `z = [x;u]`
# is represented by a type `V`.

# # Constructors
# 	GeneralKnotPoint(n::Int, m::Int, z::AbstractVector, dt, [t=0])
# 	GeneralKnotPoint(z::V, _x::SVector{n,Int}, _u::SVector{m,Int}, dt::T, t::T)
# 	KnotPoint(z::V, _x::SVector{n,Int}, _u::SVector{m,Int}, dt::T, t::T)
# """
# mutable struct GeneralKnotPoint{T,N,M,V} <: AbstractKnotPoint{T,N,M}
#     z::V
#     _x::SVector{N,Int}
#     _u::SVector{M,Int}
#     dt::T # time step
#     t::T  # total time
# end

# function GeneralKnotPoint(n::Int, m::Int, z::AbstractVector, dt::T, t=zero(T)) where T
#     _x = SVector{n}(1:n)
#     _u = SVector{m}(n .+ (1:m))
#     GeneralKnotPoint(z, _x, _u, dt, t)
# end

# function Base.copy(z::GeneralKnotPoint)
#     GeneralKnotPoint(Base.copy(z.z), z._x, z._u, z.dt, z.t)
# end

# """
# 	KnotPoint{T,n,m,nm}

# A `GeneralKnotPoint` whose stacked vector `z = [x;u]` is represented by an `SVector{nm,T}`
# where `nm = n+m`.

# # Setters
# Use the following methods to set values in a `KnotPoint`:
# ```julia
# set_state!(z::KnotPoint, x)
# set_control!(z::KnotPoint, u)
# z.t = t
# z.dt = dt
# ```

# # Constructors
# ```julia
# KnotPoint(x, u, dt, [t=0.0])
# KnotPoint(x, m, [t=0.0])  # for terminal knot point
# ```
# """
# const KnotPoint{T,N,M,NM} = GeneralKnotPoint{T,N,M,SVector{NM,T}} where {T,N,M,NM}

# function KnotPoint(z::V, ix::SVector{n,Int}, iu::SVector{m,Int}, dt::T, t::T) where {n,m,T,V}
#     GeneralKnotPoint{T,n,m,V}(z, ix, iu, dt, t)
# end

# function KnotPoint(x::AbstractVector, u::AbstractVector, dt::Float64, t=0.0)
#     n = length(x)
#     m = length(u)
#     xinds = ones(Bool, n+m)
#     xinds[n+1:end] .= 0
#     _x = SVector{n}(1:n)
#     _u = SVector{m}(n .+ (1:m))
#     z = SVector{n+m}([x;u])
#     KnotPoint(z, _x, _u, dt, t)
# end

# # Constructor for terminal time step
# function KnotPoint(x::AbstractVector, m::Int, t=0.0)
#     u = zeros(m)
#     KnotPoint(x, u, 0., t)
# end

# set_state!(z::KnotPoint, x) = z.z = [x; control(z)]
# set_control!(z::KnotPoint, u) = z.z = [state(z); u]
# set_z!(z::KnotPoint, z_) = z.z = z_


# """
# 	StaticKnotPoint{T,n,m,nm} <: AbstractKnotPoint{T,n,m}

# An immutable `AbstractKnotPoint` whose stacked vector is represented by an `SVector{nm,T}`
# where `nm = n+m`. Since `isbits(z::StaticKnotPoint) = true`, these can be created very
# efficiently and with zero allocations.

# # Constructors

# 	StaticKnotPoint(z::SVector{nm}, _x::SVector{n,Int}, _u::SVector{m,Int}, dt::Float64, t::Float64)
# 	StaticKnotPoint(x::SVector{n}, u::SVector{m}, [dt::Real=0.0, t::Real=0.0])
# 	StaticKnotPoint(z0::AbstractKnotPoint, z::AbstractVector)

# where the last constructor uses another `AbstractKnotPoint` to create a `StaticKnotPoint`
# using the stacked state-control vector `z`. If `length(z) == n`, the constructor will
# automatically append `m` zeros.

# """
# struct StaticKnotPoint{T,N,M,NM} <: AbstractKnotPoint{T,N,M}
#     z::SVector{NM,T}
#     _x::SVector{N,Int}
#     _u::SVector{M,Int}
#     dt::Float64 # time step
#     t::Float64  # total time
# end

# function StaticKnotPoint(x::SVector{n}, u::StaticVector{m}, dt=0.0, t=0.0) where {n,m}
#     ix = SVector{n}(1:n)
#     iu = n .+ SVector{m}(1:m)
#     StaticKnotPoint([x; u], ix, iu, Float64(dt), Float64(t))
# end

# function StaticKnotPoint(z0::AbstractKnotPoint{T,n,m}, z::AbstractVector=z0.z) where {T,n,m}
# 	if length(z) == n
# 		z = [SVector{n}(z); @SVector zeros(m)]
# 	end
#     StaticKnotPoint{eltype(z),n,m,n+m}(z, z0._x, z0._u, z0.dt, z0.t)
# end

# function Base.:+(z1::AbstractKnotPoint, z2::AbstractKnotPoint)
# 	StaticKnotPoint(z1.z + z2.z, z1._x, z1._u, z1.dt, z1.t)
# end

# function Base.:+(z1::AbstractKnotPoint, x::AbstractVector)
# 	StaticKnotPoint(z1.z + x, z1._x, z1._u, z1.dt, z1.t)
# end
# @inline Base.:+(x::AbstractVector, z1::AbstractKnotPoint) = z1 + x


# function Base.:*(a::Real, z::AbstractKnotPoint{<:Any,n,m}) where {n,m}
# 	StaticKnotPoint(z.z*a, SVector{n,Int}(z._x), SVector{m,Int}(z._u), z.dt, z.t)
# end

# @inline Base.:*(z::AbstractKnotPoint, a::Real) = a*z