export
    RBState,
    randbetween

"""
    RBState{T} <: StaticVector{13,T}

Represents the state of a rigid body in 3D space, consisting of position, orientation, linear
    velocity and angular velocity, respresented as a vector stacked in that order, with
    the rotation represented as the 4 elements of a unit quaternion.

Implements the `StaticArrays` interface so can be treated as an `SVector` with additional
    methods.

# Constructors
    RBState{T}(r, q, v, ω)
    RBState{T}(x)
    RBState(r, q, v, ω)
    RBState(x)

where `r`, `v`, and `ω` are three-dimensional vectors, `q` is either a `Rotation` or a
    four-dimenional vector representing the parameters of unit quaternion, and `x` is a
    13-dimensional vector (or tuple),

# Converting to a State Vector
An `RBState` can be converted to a state vector for a `RigidBody` using
    RBState(model::RBstate, x, [renorm=false])
"""
struct RBState{T} <: StaticVector{13,T}
    r::SVector{3,T}
    q::QuatRotation{T}
    v::SVector{3,T}
    ω::SVector{3,T}
    function RBState{T}(r, q::Rotation, v, ω) where {T}
        @assert length(r) == 3
        @assert length(v) == 3
        @assert length(ω) == 3
        new{T}(r, QuatRotation{T}(q), v, ω)
    end
    function RBState{T}(r, q::QuatRotation, v, ω) where {T}
        @assert length(r) == 3
        @assert length(v) == 3
        @assert length(ω) == 3
        new{T}(r, QuatRotation{T}(q.q, false), v, ω)  # don't normalize
    end
    @inline function RBState{T}(x::RBState) where {T}
        RBState{T}(x.r, x.q, x.v, x.ω)
    end
end

function RBState(r::AbstractVector, q::Rotation, v::AbstractVector, ω::AbstractVector)
    T = promote_type(eltype(r), eltype(q), eltype(v), eltype(ω))
    RBState{T}(r, q, v, ω)
end

function RBState(r::AbstractVector, q::Quaternion{TQ}, v::AbstractVector, ω::AbstractVector) where {TQ}
    T = promote_type(eltype(r), TQ, eltype(v), eltype(ω))
    RBState{T}(r, QuatRotation(q, false), v, ω)
end

@inline function RBState(r::AbstractVector, q::AbstractVector, v::AbstractVector, ω::AbstractVector)
    RBState(r, QuatRotation(q, false), v, ω)
end

@inline RBState(x::RBState) = x

@generated function RBState(::Type{R}, x::AbstractVector) where {R<:Rotation{3}}
    ir = SA[1, 2, 3]
    iq = SA[4, 5, 6]
    iv = SA[7, 8, 9]
    iω = SA[10, 11, 12]
    if R <: QuatRotation
        iq = SA[4, 5, 6, 7]
        iv = iv .+ 1
        iω = iω .+ 1
    end
    quote
        q = QuatRotation(R(x[$iq]))
        RBState(x[$ir], q, x[$iv], x[$iω])
    end
end


# Static Arrays interface
function (::Type{RB})(x::NTuple{13}) where {RB<:RBState}
    RB(
        SA[x[1], x[2], x[3]],
        QuatRotation(x[4], x[5], x[6], x[7], false),
        SA[x[8], x[9], x[10]],
        SA[x[11], x[12], x[13]]
    )
end
Base.@propagate_inbounds function Base.getindex(x::RBState, i::Int)
    if i < 4
        x.r[i]
    elseif i < 8
        Rotations.params(x.q)[i-3]
    elseif i < 11
        x.v[i-7]
    else
        x.ω[i-10]
    end
end
Base.Tuple(x::RBState) = (
    x.r[1], x.r[2], x.r[3],
    x.q.w, x.q.x, x.q.y, x.q.z,
    x.v[1], x.v[2], x.v[3],
    x.ω[1], x.ω[2], x.ω[3]
)

"""
    renorm(x::RBState)

Re-normalize the unit quaternion.
"""
@inline renorm(x::RBState) = RBState(x.r, x.q / norm(x.q.q), x.v, x.ω)

"""
    position(x::RBState)
    position(model::RigidBody, x::AbstractVector)

Return the 3-dimensional position of an rigid body as a `SVector{3}`.
"""
@inline Base.position(x::RBState) = x.r

"""
    orientation(x::RBState)
    orientation(model::RigidBody, x::AbstractVector)

Return the 3D orientation of a rigid body. Returns a `Rotations.Rotation{3}`.
"""
@inline orientation(x::RBState) = x.q

"""
    angular_velocity(x::RBState)
    angular_velocity(model::RigidBody, x::AbstractVector)

Return the 3D linear velocity of a rigid body as a `SVector{3}`.
"""
@inline angular_velocity(x::RBState) = x.ω

"""
    linear_velocity(x::RBState)
    linear_velocity(model::RigidBody, x::AbstractVector)

Return the 3D linear velocity of a rigid body as a `SVector{3}`.
"""
@inline linear_velocity(x::RBState) = x.v


function Base.isapprox(x1::RBState, x2::RBState; kwargs...)
    isapprox(x1.r, x2.r; kwargs...) &&
        isapprox(x1.v, x2.v; kwargs...) &&
        isapprox(x1.ω, x2.ω; kwargs...) &&
        isapprox(principal_value(x1.q), principal_value(x2.q); kwargs...)
end

"""
    +(::RBState, ::RBState)

Add two rigid body states, which adds the position, linear and angular velocities, and
    composes the orientations.
"""
function Base.:+(s1::RBState, s2::RBState)
    RBState(s1.r + s2.r, s1.q * s2.q, s1.v + s2.v, s1.ω + s2.ω)
end

"""
    -(::RBState, ::RBState)

Substract two rigid body states, which substracts the position, linear and angular velocities,
    and composes the inverse of the second orientation with the first, i.e. `inv(q2)*q1`.
"""
function Base.:-(s1::RBState, s2::RBState)
    RBState(s1.r - s2.r, s2.q \ s1.q, s1.v - s2.v, s1.ω - s2.ω)
end

"""
    ⊖(::RBState, ::RBState)

Compute the 12-dimensional error state, calculated by substracting thep position, linear,
    and angular velocities, and using `Rotations.rotation_error` for the orientation.
"""
function Rotations.:⊖(s1::RBState, s2::RBState, rmap=CayleyMap())
    dx = s1.r - s2.r
    dq = Rotations.rotation_error(s1.q, s2.q, rmap)
    dv = s1.v - s2.v
    dw = s1.ω - s2.ω
    @SVector [dx[1], dx[2], dx[3], dq[1], dq[2], dq[3],
        dv[1], dv[2], dv[3], dw[1], dw[2], dw[3]]
end

"""
    ⊕(x::RBState, dx::StaticVector{12}, rmap=CayleyMap())

Add the state error `dx` to the rigid body state `x` using the map `rmap`. Simply adds the
vector states, and computes the orientation with `Rotations.add_error(x, dx, rmap)`
"""
function Rotations.:⊕(s1::RBState, dx::StaticVector{12}, rmap=CayleyMap())
    dr = SA[dx[1], dx[2], dx[3]]
    dq = SA[dx[4], dx[5], dx[6]]
    dv = SA[dx[7], dx[8], dx[9]]
    dω = SA[dx[10], dx[11], dx[12]]
    q = Rotations.add_error(s1.q, Rotations.RotationError(dq, rmap))
    RBState(s1.r + dr, q, s1.v + dv, s1.ω + dω)
end

Base.zero(s1::RBState) = zero(RBState)
@inline Base.zero(::Type{<:RBState}) = zero(RBState{Float64})
function Base.zero(::Type{<:RBState{T}}) where {T}
    RBState((@SVector zeros(T, 3)), QuatRotation{T}(I),
        (@SVector zeros(T, 3)), (@SVector zeros(T, 3)))
end

function Base.rand(::Type{RBState})
    RBState(rand(3), rand(QuatRotation), rand(3), rand(3))
end

function randbetween(xmin::RBState, xmax::RBState)
    dx = xmax - xmin
    RBState(
        xmin.r .+ rand(3) .* dx.r,
        # rand(QuatRotation),
        expm((@SVector randn(3)) * rand() * deg2rad(170)),
        xmin.v .+ rand(3) .* dx.v,
        xmin.ω .+ rand(3) .* dx.ω
    )
end
