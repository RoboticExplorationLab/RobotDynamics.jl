"""
	AbstractKnotPoint{Nx,Nu,V,T}

Stores the states, controls, time, and time step at a single point along a trajectory of a
forced dynamical system with `n` states and `m` controls.

# Interface
All instances of `AbstractKnotPoint` should support the following methods:

| **Required methods**  |    | **Brief description**  |
|:--------------------- |:-- |:---------------------- |
| `state_dim(z)` | | State vector dimension |
| `control_dim(z)` | | Control vector dimension |
| `get_data(z)::V` | | Get the vector of concatenated states and controls |
| **Optional Methods** | **Default definition** | **Brief description** | 
| `time(z)` | `z.t` | Get the time |
| `timestep(z)` | `z.dt` | Get the time step |
| `settime!(z, t)` | `z.t = t` | Set the time |
| `settimestep!(z, dt)` | `z.dt = dt` | Set the time step |
| `is_terminal(z)` | `timestep(z) === zero(datatype(z))` |

# Methods
Given the above interface, the following methods are defined for an `AbstractKnotPoint`:

* `dims(z)`
* [`getstate`](@ref)
* [`getcontrol`](@ref)
* [`state`](@ref)
* [`control`](@ref)
* [`state`](@ref)
* [`setdata!`](@ref)
* [`setstate!`](@ref)
* [`setcontrol!`](@ref)
* [`is_terminal`](@ref)
* [`datatype`](@ref)
* [`vectype`](@ref)
"""
abstract type AbstractKnotPoint{Nx,Nu,V,T} <: AbstractVector{T} end

dims(z::AbstractKnotPoint) = (state_dim(z), control_dim(z))

"""
    getstate(z, v)

Extract the state vector from a vector `v`. Returns a view into the matrix by default, 
or an `SVector` if the knot point uses `SVector`s.
"""
getstate(z::AbstractKnotPoint, v) = view(v, 1:state_dim(z))

"""
    getcontrol(z, v)

Extract the state vector from a vector `v`. Returns a view into the matrix by default, 
or an `SVector` if the knot point uses `SVector`s. If the knot point is a terminal 
knot point, it will return an empty view or an `SVector` of zeros.
"""
function getcontrol(z::AbstractKnotPoint, v)
    if is_terminal(z)
        return view(v, length(v):length(v)-1)
    else
        n,m = dims(z)
        return view(v, n+1:n+m) 
    end
end

getstate(z::AbstractKnotPoint{Nx,Nu,<:SVector}, v) where {Nx,Nu} = v[SVector{Nx}(1:Nx)]
getcontrol(z::AbstractKnotPoint{Nx,Nu,<:SVector}, v) where {Nx,Nu} = !is_terminal(z) * v[SVector{Nu}(Nx+1:Nx+Nu)]

"""
    state(z)

Get the state vector. Returns either a view or an `SVector`.
"""
state(z::AbstractKnotPoint) = getstate(z, getdata(z))

"""
    control(z)

Get the control vector. Returns either a view of an `SVector`. For a terminal knot point
it will return an empty view or an `SVector` of zeros.
"""
control(z::AbstractKnotPoint) = getcontrol(z, getdata(z))

"""
    setdata!(z, v)

Set the data vector, or the concatenated vector of states and controls.
"""
setdata!(z::AbstractKnotPoint, v) = z.z .= v
setdata!(z::AbstractKnotPoint{<:Any,<:Any,<:SVector}, v) = z.z = v

"""
    setstate!(z, x)

Set the state vector for an [`AbstractKnotPoint`](@ref).
"""
setstate!(z::AbstractKnotPoint, x) = state(z) .= x
setstate!(z::AbstractKnotPoint{Nx,<:Any,<:SVector}, x) where Nx = setdata!(z, [SVector{Nx}(x); control(z)])

"""
    setcontrol!(z, x)

Set the control vector for an [`AbstractKnotPoint`](@ref).
"""
setcontrol!(z::AbstractKnotPoint, u) = control(z) .= u
setcontrol!(z::AbstractKnotPoint{<:Any,Nu,<:SVector}, u) where Nu = setdata!(z, [state(z); SVector{Nu}(u)])

settime!(z::AbstractKnotPoint, t) = z.t = t
settimestep!(z::AbstractKnotPoint, dt) = z.dt = dt

time(z::AbstractKnotPoint) = getparams(z).t 
timestep(z::AbstractKnotPoint) = getparams(z).dt 

"""
    is_terminal(z)

Determines if the knot point `z` is a terminal knot point with no 
controls, only state information. By default a knot point is assumed to be a 
terminal knot point if the time step is zero. By convention, if the last 
knot point has control values assigned (for example, when doing first-order 
hold on the controls), the final time step is set to infinity instead of zero.
"""
is_terminal(z::AbstractKnotPoint) = timestep(z) === zero(datatype(z))

"""
    datatype(x)

Get the numeric data type used by the object `x`. Typically a floating point data type.
"""
datatype(z::AbstractKnotPoint{<:Any,<:Any,<:Any,T}) where T = T

"""
    vectype(x)

Get the vector type used by the object `x`. Used to allow either static or 
dynamic arrays in structs such as [`AbstractKnotPoint`](@ref).
"""
vectype(z::AbstractKnotPoint{<:Any,<:Any,V}) where V = V

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
                @assert Nx + Nu == length(z)
                new{Nx,Nu,V,eltype(V)}(z, t, dt, Nx, Nu)
            end
            function $name{Nx,Nu}(x::V, u::V, t,dt) where {Nx,Nu,V} 
                Nx != Any && @assert Nx == length(x)
                Nu != Any && @assert Nu == length(u) 
                new{Nx,Nu,V,eltype(V)}([x;u], t, dt, Nx, Nu)
            end
            function $name(n::Integer, m::Integer, z::V, t, dt) where V
                @assert n > 0
                @assert m > 0
                @assert n + m == length(z)
                new{Any,Any,V,eltype(V)}(z, t, dt, n, m)
            end
            function $name{Nx,Nu}(n::Integer, m::Integer, z::V, t, dt) where {Nx,Nu,V} 
                Nx != Any && @assert Nx == n
                Nu != Any && @assert Nu == m
                @assert n + m == length(z)
                new{Nx,Nu,V,eltype(V)}(z, t, dt, n, m)
            end
        end
        function $name(x::StaticVector{Nx}, u::StaticVector{Nu}, t, dt) where {Nx, Nu}
            $name{Nx,Nu}([x;u], t, dt)
        end
        function $name(x::AbstractVector, u::AbstractVector, t, dt)
            $name(length(x), length(u), [x;u], t, dt)
        end
        function Base.copy(z::$name{Nx,Nu}) where {Nx,Nu}
            $name{Nx,Nu}(z.n, z.m, z.z, z.t, z.dt)
        end

        state_dim(z::$name{Any}) = z.n
        control_dim(z::$name{<:Any,Any}) = z.m
        state_dim(z::$name{Nx}) where Nx = Nx 
        control_dim(z::$name{<:Any,Nu}) where Nu = Nu 
        vectype(::$name{<:Any,<:Any,V}) where V = V
        vectype(::Type{<:$name{<:Any,<:Any,V}}) where V = V

        @inline getparams(z::$name) = (t=z.t, dt=z.dt)
        @inline getdata(z::$name) = z.z
        function Base.copyto!(dest::$name, src::$name)
            if ismutabletype($name)
                dest.t = src.t
                dest.dt = src.dt
            end
            copyto!(dest.z, src.z)
            dest
        end

        # Array interface
        Base.similar(z::$name{Nx,Nu,V,T}, ::Type{S}, dims::Dims) where {Nx,Nu,V,T,S} = 
            KnotPoint{Nx,Nu}(z.n, z.m, similar(z.z, S, dims), z.t, z.dt)
        Base.similar(z::$name{Nx,Nu,<:SVector,T}, ::Type{S}, dims::Dims) where {Nx,Nu,T,S} = 
            KnotPoint{Nx,Nu}(z.n, z.m, similar(MVector{Nx+Nu,S}), z.t, z.dt)
    end
    # Set struct mutability
    if mutable
        struct_expr = expr.args[2]
        @assert struct_expr.head == :struct
        struct_expr.args[1] = mutable
    end
    eval(expr)
end

function (::Type{<:AbstractKnotPoint{Nx,Nu}})(z, t, dt) where {Nx,Nu}
    KnotPoint{Nx,Nu}(z, t, dt)
end

function StaticKnotPoint(z::AbstractKnotPoint{Nx,Nu}, v::AbstractVector) where {Nx,Nu}
    StaticKnotPoint{Nx,Nu}(state_dim(z), control_dim(z), v, time(z), timestep(z))
end
@inline StaticKnotPoint(z::AbstractKnotPoint) = StaticKnotPoint(z, getdata(z)) 
@inline setdata(z::StaticKnotPoint, v) = StaticKnotPoint(z, v)
setstate(z::StaticKnotPoint, x) = setdata(z, [x; control(z)])
setcontrol(z::StaticKnotPoint, u) = setdata(z, [state(z); u])

function Base.:*(c::Real, z::KP) where {KP<:AbstractKnotPoint}
    KP(getdata(z)*c, getparams(z)...)
end

"""
    KnotPoint{Nx,Nu,V,T}

A mutable [`AbstractKnotPoint`](@ref) with `Nx` states, `Nu` controls, stored using 
a vector type `V` with data type `T`. Since the struct is mutable, the time, timestep, 
and data can all be changed, which can be very efficient when the data being stored 
as an `SVector`.

# Constructors

    KnotPoint{n,m}(v, t, dt)
    KnotPoint{n,m}(x, u, t, dt)
    KnotPoint{n,m}(n, m, v, t, dt) 
    KnotPoint(n, m, v, t, dt)       # create a KnotPoint{Any,Any}
    KnotPoint(x, u, t, dt)

The last method will create a `KnotPoint{Any,Any}` if `x` and `u` are not `StaticVector`s.

The vector type `V` can be queried using `vectype(z)`.
"""
KnotPoint

"""
    StaticKnotPoint

A static version of [`KnotPoint`](@ref). Uses all of the same methods and constructors, 
but also adds the following methods:

    StaticKnotPoint(z, v)

which creates a new `StaticKnotPoint` using the information from the `AbstractKnotPoint` z,
but using data from `v`. Useful for creating temporary knot points from existing ones 
without any runtime allocations.

The following methods are similar to their mutable versions, but create a new 
`StaticKnotPoint`:

    setdata
    setstate
    setcontrol
"""
StaticKnotPoint
