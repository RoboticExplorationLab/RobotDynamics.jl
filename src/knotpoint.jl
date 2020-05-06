export
    KnotPoint,
    state,
    control

abstract type AbstractKnotPoint{T,N,M} end

"""
	GeneralKnotPoint{T,N,M,V} <: AbstractKnotPoint{T,N,M}

Stores critical information corresponding to each knot point in the trajectory optimization
problem, including the state and control values, as well as the time and time step length.

# Getters
Use the following methods to access values from a `KnotPoint`:
```julia
x  = state(z::KnotPoint)    # returns the n-dimensional state as a SVector
u  = control(z::KnotPoint)  # returns the m-dimensional control vector as a SVector
t  = z.t                    # current time
dt = z.dt                   # time step length
```

# Setters
Use the following methods to set values in a `KnotPoint`:
```julia
set_state!(z::KnotPoint, x)
set_control!(z::KnotPoint, u)
z.t = t
z.dt = dt
```

# Constructors
```julia
KnotPoint(x, u, dt, t=0.0)
KnotPoint(x, m, t=0.0)  # for terminal knot point
```

Use `is_terminal(z::KnotPoint)` to determine if a `KnotPoint` is a terminal knot point (e.g.
has no time step length and z.t == tf).
"""
mutable struct GeneralKnotPoint{T,N,M,V} <: AbstractKnotPoint{T,N,M}
    z::V
    _x::SVector{N,Int}
    _u::SVector{M,Int}
    dt::T # time step
    t::T  # total time
end

function GeneralKnotPoint(n::Int, m::Int, z::AbstractVector, dt::T, t=zero(T)) where T
    _x = SVector{n}(1:n)
    _u = SVector{m}(n .+ (1:m))
    GeneralKnotPoint(z, _x, _u, dt, t)
end

const KnotPoint{T,N,M,NM} = GeneralKnotPoint{T,N,M,SVector{NM,T}} where {T,N,M,NM}

function KnotPoint(z::V, ix::SVector{n,Int}, iu::SVector{m,Int}, dt::T, t::T) where {n,m,T,V}
    GeneralKnotPoint{T,n,m,V}(z, ix, iu, dt, t)
end

function KnotPoint(x::AbstractVector, u::AbstractVector, dt::Float64, t=0.0)
    n = length(x)
    m = length(u)
    xinds = ones(Bool, n+m)
    xinds[n+1:end] .= 0
    _x = SVector{n}(1:n)
    _u = SVector{m}(n .+ (1:m))
    z = SVector{n+m}([x;u])
    KnotPoint(z, _x, _u, dt, t)
end

# Constructor for terminal time step
function KnotPoint(x::AbstractVector, m::Int, t=0.0)
    u = zeros(m)
    KnotPoint(x, u, 0., t)
end

@inline state(z::AbstractKnotPoint) = z.z[z._x]
@inline control(z::AbstractKnotPoint) = z.z[z._u]
@inline is_terminal(z::AbstractKnotPoint) = z.dt == 0
@inline get_z(z::RobotDynamics.AbstractKnotPoint) = is_terminal(z) ? state(z) : z.z

set_state!(z::AbstractKnotPoint, x) = for i in z._x; z.z[i] = x[i]; end
set_control!(z::AbstractKnotPoint, u) = for (i,j) in enumerate(z._u); z.z[j] = u[i]; end
set_z!(z::AbstractKnotPoint, z_) = is_terminal(z) ? set_state!(z, z_) : copyto!(z.z, z_)
set_state!(z::KnotPoint, x) = z.z = [x; control(z)]
set_control!(z::KnotPoint, u) = z.z = [state(z); u]
set_z!(z::KnotPoint, z_) = z.z = z_

struct StaticKnotPoint{T,N,M,NM} <: AbstractKnotPoint{T,N,M}
    z::SVector{NM,T}
    _x::SVector{N,Int}
    _u::SVector{M,Int}
    dt::Float64 # time step
    t::Float64  # total time
end

function StaticKnotPoint(x::SVector{n,T}, u::SVector{m,T}, dt=zero(T), t=zero(T)) where {n,m,T}
    ix = SVector{n}(1:n)
    iu = n .+ SVector{m}(1:m)
    StaticKnotPoint([x; u], ix, iu, dt, t)
end

function StaticKnotPoint(z0::AbstractKnotPoint{T,n,m}, z::AbstractVector=z0.z) where {T,n,m}
	if length(z) == n
		z = [SVector{n}(z); @SVector zeros(m)]
	end
    StaticKnotPoint{eltype(z),n,m,n+m}(z, z0._x, z0._u, z0.dt, z0.t)
end

function Base.:+(z1::AbstractKnotPoint, z2::AbstractKnotPoint)
	StaticKnotPoint(z1.z + z2.z, z1._x, z1._u, z1.dt, z1.t)
end

function Base.:+(z1::AbstractKnotPoint, x::AbstractVector)
	StaticKnotPoint(z1.z + x, z1._x, z1._u, z1.dt, z1.t)
end

function Base.:*(a::Real, z::AbstractKnotPoint{<:Any,n,m}) where {n,m}
	StaticKnotPoint(z.z*a, SVector{n,Int}(z._x), SVector{m,Int}(z._u), z.dt, z.t)
end
