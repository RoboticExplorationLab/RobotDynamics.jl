export
    KnotPoint,
    state,
    control

abstract type AbstractKnotPoint{T,N,M} end

""" $(TYPEDEF)
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
mutable struct KnotPoint{T,N,M,NM} <: AbstractKnotPoint{T,N,M}
    z::SVector{NM,T}
    _x::SVector{N,Int}
    _u::SVector{M,Int}
    dt::T # time step
    t::T  # total time
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


set_state!(z::KnotPoint, x) = z.z = [x; control(z)]
set_control!(z::KnotPoint, u) = z.z = [state(z); u]

struct StaticKnotPoint{T,N,M,NM} <: AbstractKnotPoint{T,N,M}
    z::SVector{NM,T}
    _x::SVector{N,Int}
    _u::SVector{M,Int}
    dt::T # time step
    t::T  # total time
end

""" A vector of KnotPoints """
const Traj = AbstractVector{<:AbstractKnotPoint}
