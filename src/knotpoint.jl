export
    KnotPoint,
    state,
    control

"""
	AbstractKnotPoint{T,n,m}

Stores the states, controls, time, and time step at a single point along a trajectory of a
forced dynamical system with `n` states and `m` controls.

# Interface
All instances of `AbstractKnotPoint` should support the following methods:

	state(z)::StaticVector{n}     # state vector
	control(z)::StaticVector{m}   # control vector
	z.t::Real                     # time
	z.dt::Real                    # time to next point (time step)

By default, it is assumed that if `z.dt == 0` the point is the last point in the trajectory.

Alternatively, the methods `state` and `control` will be automatically defined if the
following fields are present:
- `z.z`: the stacked vector `[x;u]`
- `z._x`: the indices of the states, such that `x = z.z[z._x]`
- `z._u`: the indices of the controls, such that `x = u.z[z._u]`
"""
abstract type AbstractKnotPoint{T,N,M} end

"""
	state(::AbstractKnotPoint)

Return the `n`-dimensional state vector
"""
@inline state(z::AbstractKnotPoint) = z.z[z._x]

"""
	control(::AbstractKnotPoint)

Return the `m`-dimensional control vector
"""
@inline control(z::AbstractKnotPoint) = z.z[z._u]

"""
	is_terminal(::AbstractKnotPoint)::Bool

Determine if the knot point is the terminal knot point, which is the case when `z.dt == 0`.
"""
@inline is_terminal(z::AbstractKnotPoint) = z.dt == 0

"""
	get_z(::AbstractKnotPoint)

Returns the stacked state-control vector `z`, or just the state vector if `is_terminal(z) == true`.
"""
@inline get_z(z::AbstractKnotPoint) = is_terminal(z) ? state(z) : z.z

"""
	set_state!(z::AbstractKnotPoint, x::AbstractVector)

Set the state in `z` to `x`.
"""
set_state!(z::AbstractKnotPoint, x) = for i in z._x; z.z[i] = x[i]; end

"""
	set_control!(z::AbstractKnotPoint, u::AbstractVector)

Set the controls in `z` to `u`.
"""
set_control!(z::AbstractKnotPoint, u) = for (i,j) in enumerate(z._u); z.z[j] = u[i]; end

"""
	set_z!(z::AbstractKnotPoint, z_::AbstractVector)

Set both the states and controls in `z` from the stacked state-control vector `z_`, unless
`is_terminal(z)`, in which case `z_` is assumed to be the terminal states.
"""
set_z!(z::AbstractKnotPoint, z_) = is_terminal(z) ? set_state!(z, z_) : copyto!(z.z, z_)

"""
	GeneralKnotPoint{T,n,m,V} <: AbstractKnotPoint{T,n,m}

A mutable instantiation of the `AbstractKnotPoint` interface where the joint vector `z = [x;u]`
is represented by a type `V`.

# Constructors
	GeneralKnotPoint(n::Int, m::Int, z::AbstractVector, dt, [t=0])
	GeneralKnotPoint(z::V, _x::SVector{n,Int}, _u::SVector{m,Int}, dt::T, t::T)
	KnotPoint(z::V, _x::SVector{n,Int}, _u::SVector{m,Int}, dt::T, t::T)
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

"""
	KnotPoint{T,n,m,nm}

A `GeneralKnotPoint` whose stacked vector `z = [x;u]` is represented by an `SVector{nm,T}`
where `nm = n+m`.

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
KnotPoint(x, u, dt, [t=0.0])
KnotPoint(x, m, [t=0.0])  # for terminal knot point
```
"""
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

set_state!(z::KnotPoint, x) = z.z = [x; control(z)]
set_control!(z::KnotPoint, u) = z.z = [state(z); u]
set_z!(z::KnotPoint, z_) = z.z = z_

"""
	StaticKnotPoint{T,n,m,nm} <: AbstractKnotPoint{T,n,m}

An immutable `AbstractKnotPoint` whose stacked vector is represented by an `SVector{nm,T}`
where `nm = n+m`. Since `isbits(z::StaticKnotPoint) = true`, these can be created very
efficiently and with zero allocations.

# Constructors

	StaticKnotPoint(z::SVector{nm}, _x::SVector{n,Int}, _u::SVector{m,Int}, dt::Float64, t::Float64)
	StaticKnotPoint(x::SVector{n}, u::SVector{m}, [dt::Real=0.0, t::Real=0.0])
	StaticKnotPoint(z0::AbstractKnotPoint, z::AbstractVector)

where the last constructor uses another `AbstractKnotPoint` to create a `StaticKnotPoint`
using the stacked state-control vector `z`. If `length(z) == n`, the constructor will
automatically append `m` zeros.

"""
struct StaticKnotPoint{T,N,M,NM} <: AbstractKnotPoint{T,N,M}
    z::SVector{NM,T}
    _x::SVector{N,Int}
    _u::SVector{M,Int}
    dt::Float64 # time step
    t::Float64  # total time
end

function StaticKnotPoint(x::SVector{n}, u::StaticVector{m}, dt=0.0, t=0.0) where {n,m}
    ix = SVector{n}(1:n)
    iu = n .+ SVector{m}(1:m)
    StaticKnotPoint([x; u], ix, iu, Float64(dt), Float64(t))
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
@inline Base.:+(x::AbstractVector, z1::AbstractKnotPoint) = z1 + x


function Base.:*(a::Real, z::AbstractKnotPoint{<:Any,n,m}) where {n,m}
	StaticKnotPoint(z.z*a, SVector{n,Int}(z._x), SVector{m,Int}(z._u), z.dt, z.t)
end

@inline Base.:*(z::AbstractKnotPoint, a::Real) = a*z
