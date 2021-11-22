"""
	AbstractKnotPoint{Nx,Nu,V,T}

Stores the states, controls, time, and time step at a single point along a trajectory of a
forced dynamical system with `n` states and `m` controls.

# Interface
All instances of `AbstractKnotPoint` should support the following methods:

	state_dim(z)::Integer         # state vector dimension
	control_dim(z)::Integer       # control vector dimension
	getparams(z)                  # arbitrary parameters, usually the time and time step 
	getdata(z)::V                 # the underlying vector
"""
abstract type AbstractKnotPoint{Nx,Nu,V,T} <: AbstractVector{T} end

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

is_terminal(z::AbstractKnotPoint) = getparams(z).dt ≈ 0.0 

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
            function $name{Nx,Nu}(x::V, u::V, t,dt) where {Nx,Nu,V} 
                @assert Nx > 0
                @assert Nu > 0
                new{Nx,Nu,V,eltype(V)}([x;u], t, dt, Nx, Nu)
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
            $name{Nx,Nu}([x;u], t, dt)
        end
        function $name(x::AbstractVector, u::AbstractVector, t, dt)
            $name(length(x), length(u), [x;u], t, dt)
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

function (::Type{<:KnotPoint{Nx,Nu}})(z, t, dt) where {Nx,Nu}
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