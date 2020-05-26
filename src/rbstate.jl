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
    13-dimensional vector (or tuple).
"""
struct RBState{T} <: StaticVector{13,T}
    r::SVector{3,T}
    q::UnitQuaternion{T}
    v::SVector{3,T}
    ω::SVector{3,T}
    function RBState{T}(r, q::Rotation, v, ω) where T
        @assert length(r) == 3
        @assert length(v) == 3
        @assert length(ω) == 3
        new{T}(r, UnitQuaternion{T}(q), v, ω)
    end
    @inline function RBState{T}(x::RBState) where T
        RBState{T}(x.r, x.q, x.v, x.ω)
    end
end

function RBState(r::AbstractVector, q::Rotation, v::AbstractVector, ω::AbstractVector)
    T = promote_type(eltype(r), eltype(q), eltype(v), eltype(ω))
    RBState{T}(r, q, v, ω)
end

@inline function RBState(r::AbstractVector, q::AbstractVector, v::AbstractVector, ω::AbstractVector)
    RBState(r, UnitQuaternion(q, false), v, ω)
end

@inline RBState(x::RBState) = x

# Static Arrays interface
function (::Type{RB})(x::NTuple{13}) where RB <: RBState
    RB(
        SA[x[1], x[2], x[3]],
        UnitQuaternion(x[4], x[5], x[6], x[7], false),
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
@inline renorm(x::RBState) = RBState(x.r, UnitQuaternion(x.q), x.v, x.ω)

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
    RBState(s1.r+s2.r, s1.q*s2.q, s1.v+s2.v, s1.ω+s2.ω)
end

"""
    +(::RBState, ::RBState)

Substract two rigid body states, which substracts the position, linear and angular velocities,
    and composes the inverse of the second orientation with the first, i.e. `inv(q2)*q1`.
"""
function Base.:-(s1::RBState, s2::RBState)
    RBState(s1.r-s2.r, s2.q\s1.q, s1.v-s2.v, s1.ω-s2.ω)
end

"""
    ⊖(::RBState, ::RBState)

Compute the 12-dimensional error state, calculated by substracting thep position, linear,
    and angular velocities, and using `Rotations.rotation_error` for the orientation.
"""
function Rotations.:⊖(s1::RBState, s2::RBState, rmap=ExponentialMap)
    dx = s1.r-s2.r
    dq = Rotations.rotation_error(s1.q, s2.q, rmap)
    dv = s1.v-s2.v
    dw = s1.ω-s2.ω
    @SVector [dx[1], dx[2], dx[3], dq[1], dq[2], dq[3],
              dv[1], dv[2], dv[3], dw[1], dw[2], dw[3]]
end

Base.zero(s1::RBState) = zero(RBState)
function Base.zero(::Type{<:RBState})
    RBState((@SVector zeros(3)), UnitQuaternion(I),
        (@SVector zeros(3)), (@SVector zeros(3)))
end

function Base.rand(::Type{RBState})
    RBState(rand(3), rand(UnitQuaternion), rand(3), rand(3))
end

function randbetween(xmin::RBState, xmax::RBState)
    dx = xmax - xmin
    RBState(
        xmin.r .+ rand(3) .* dx.r,
        # rand(UnitQuaternion),
        expm((@SVector randn(3))*rand()*deg2rad(170)),
        xmin.v .+ rand(3) .* dx.v,
        xmin.ω .+ rand(3) .* dx.ω
    )
end
