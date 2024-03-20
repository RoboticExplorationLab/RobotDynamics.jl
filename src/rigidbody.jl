"""
	RigidBody{R<:Rotation} <: LieGroupModel

Abstraction of a dynamical system with free-body dynamics, with a 12 or 13-dimensional state
vector: `[p; q; v; ω]`
where `p` is the 3D position, `q` is the 3 or 4-dimension attitude representation, `v` is the
3D linear velocity, and `ω` is the 3D angular velocity.

# Interface
Any single-body system can leverage the `RigidBody` type by inheriting from it and defining the
following interface:

```julia
forces(::MyRigidBody, x, u, [t])  # return the forces in the world frame
moments(::MyRigidBody, x, u, [t]) # return the moments in the body frame
inertia(::MyRigidBody, x, u) # return the 3x3 inertia matrix
mass(::MyRigidBody, x, u)  # return the mass as a real scalar
```

Instead of defining `forces` and `moments` you can define the higher-level `wrenches` function

	wrenches(model::MyRigidbody, x, u, t)

# Rotation Parameterization
A `RigidBody` model must specify the rotational representation being used. Any `Rotations.Rotation{3}`
can be used, but we suggest one of the following:
* `QuatRotation`
* `MRP`
* `RodriguesParam`

# Working with state vectors for a `RigidBody`
Several methods are provided for working with the state vectors for a `RigidBody`.
Also see the documentation for [`RBState`](@ref) which provides a unified representation
for working with states for rigid bodies, which can be easily converted to and 
from the state vector representation for the given model.

- [`Base.position`](@ref)
- [`orientation`](@ref)
- [`linear_velocity`](@ref)
- [`angular_velocity`](@ref)
- [`build_state`](@ref)
- [`parse_state`](@ref)
- [`gen_inds`](@ref)
- [`flipquat`](@ref)
"""
abstract type RigidBody{R<:Rotation} <: LieGroupModel end

LieState(::RigidBody{R}) where {R} = LieState(R, (3, 6))

function Base.rand(model::RigidBody{D}) where {D}
    n, m = dims(model)
    r = @SVector rand(3)
    q = rand(D)
    v = @SVector rand(3)
    ω = @SVector rand(3)
    x = build_state(model, r, q, v, ω)
    u = @SVector rand(m)  # NOTE: this is type unstable
    return x, u
end

function Base.zeros(model::RigidBody{D}) where {D}
    n, m = dims(model)
    r = @SVector zeros(3)
    q = one(D)
    v = @SVector zeros(3)
    ω = @SVector zeros(3)
    x = build_state(model, r, q, v, ω)
    u = @SVector zeros(m)  # NOTE: this is type unstable
    return x, u
end

@inline rotation_type(::RigidBody{D}) where {D} = D

"""
    gen_inds(model::RigidBody)

Generate a `NamedTuple` containing the indices of the position (`r`), orientation (`q`),
linear velocity (`v`), and angular velocity (`ω`) from the state vector for `model`.
"""
@generated function gen_inds(model::RigidBody{R}) where {R}
    iF = SA[1, 2, 3]
    iM = SA[4, 5, 6]
    ir, iq, iv, iω = SA[1, 2, 3], SA[4, 5, 6], SA[7, 8, 9], SA[10, 11, 12]
    if R <: QuatRotation
        iq = push(iq, 7)
        iv = iv .+ 1
        iω = iω .+ 1
    end
    quote
        m = control_dim(model)
        iu = $iω[end] .+ SVector{m}(1:m)
        return (r=$ir, q=$iq, v=$iv, ω=$iω, u=iu)
    end
end

# Getters
@inline Base.position(model::RigidBody, x) = x[gen_inds(model).r]
@inline linear_velocity(model::RigidBody, x) = x[gen_inds(model).v]
@inline angular_velocity(model::RigidBody, x) = x[gen_inds(model).ω]

function orientation(model::RigidBody{<:QuatRotation}, x::AbstractVector,
    renorm=false)
    q = QuatRotation(x[4], x[5], x[6], x[7], renorm)
    return q
end
for rot in [RodriguesParam, MRP, RotMatrix, RotationVec, AngleAxis]
    @eval orientation(model::RigidBody{<:$rot}, x::AbstractVector, renorm=false) = ($rot)(x[4], x[5], x[6])
end

"""
    flipquat(model, x)

Flips the quaternion sign for a `RigidBody{<:QuatRotation}`.
"""
function flipquat(model::RigidBody{<:QuatRotation}, x)
    return @SVector [x[1], x[2], x[3], -x[4], -x[5], -x[6], -x[7],
        x[8], x[9], x[10], x[11], x[12], x[13]]
end

"""
    parse_state(model::RigidBody{R}, x, renorm=false)

Return the position, orientation, linear velocity, and angular velocity as separate vectors.
The orientation will be of type `R`. If `renorm=true` and `R <: QuatRotation` the quaternion
will be renormalized.
"""
function parse_state(model::RigidBody, x, renorm=false)
    r = position(model, x)
    p = orientation(model, x, renorm)
    v = linear_velocity(model, x)
    ω = angular_velocity(model, x)
    return r, p, v, ω
end

function RBState(model::RigidBody, x, renorm=false)
    RBState(parse_state(model, x, renorm)...)
end

function RBState(model::RigidBody, x::RBState, renorm=false)
    if renorm
        renorm(x)
    else
        x
    end
end

@inline RBState(model::RigidBody, z::AbstractKnotPoint, renorm=false) = RBState(model, state(z), renorm)


"""
    build_state(model::RigidBody{R}, x::RBState) where R
    build_state(model::RigidBody{R}, x::AbstractVector) where R
    build_state(model::RigidBody{R}, r, q, v, ω) where R

Build the state vector for `model` using the `RBState` `x`. If `R <: QuatRotation` this
    returns `x` cast as an `SVector`, otherwise it will convert the quaternion in `x` to
    a rotation of type `R`.

Also accepts as arguments any arguments that can be passed to the constructor of `RBState`.
"""
@inline build_state(model::RigidBody{<:QuatRotation}, x::RBState) = SVector(x)
function build_state(model::RigidBody{R}, x::RBState) where {R<:Rotation}
    r = position(x)
    q = Rotations.params(R(orientation(x)))
    v = linear_velocity(x)
    ω = angular_velocity(x)
    SA[
        r[1], r[2], r[3],
        q[1], q[2], q[3],
        v[1], v[2], v[3],
        ω[1], ω[2], ω[3],
    ]
end

function build_state(model::RigidBody{R}, args...) where {R<:Rotation}
    x_ = RBState(args...)
    build_state(model, x_)
end

@inline function build_state(model::RigidBody,
    r::AbstractVector, q::AbstractVector, v::AbstractVector, ω::AbstractVector)
    @assert length(q) == 3
    SA[
        r[1], r[2], r[3],
        q[1], q[2], q[3],
        v[1], v[2], v[3],
        ω[1], ω[2], ω[3],
    ]
end

@inline function build_state(model::RigidBody{<:QuatRotation},
    r::AbstractVector, q::AbstractVector, v::AbstractVector, ω::AbstractVector)
    @assert length(q) == 4
    SA[
        r[1], r[2], r[3],
        q[1], q[2], q[3], q[4],
        v[1], v[2], v[3],
        ω[1], ω[2], ω[3],
    ]
end

function fill_state(model::RigidBody{<:QuatRotation}, x::Real, q::Real, v::Real, ω::Real)
    @SVector [x, x, x, q, q, q, q, v, v, v, ω, ω, ω]
end

function fill_state(model::RigidBody, x::Real, q::Real, v::Real, ω::Real)
    @SVector [x, x, x, q, q, q, v, v, v, ω, ω, ω]
end

############################################################################################
#                                DYNAMICS
############################################################################################
function dynamics(model::RigidBody, x, u, t=0)

    r, q, v, ω = parse_state(model, x)

    ξ = wrenches(model, x, u, t)
    F = SA[ξ[1], ξ[2], ξ[3]]  # forces in world frame
    τ = SA[ξ[4], ξ[5], ξ[6]]  # torques in body frame
    m = mass(model)
    J = inertia(model)
    Jinv = inertia_inv(model)

    qdot = Rotations.kinematics(q, ω)
    if velocity_frame(model) == :world
        xdot = v
        vdot = F ./ m
    elseif velocity_frame(model) == :body
        xdot = q * v
        vdot = q \ (F ./ m) - ω × v
    end
    ωdot = Jinv * (τ - ω × (J * ω))

    build_state(model, xdot, qdot, vdot, ωdot)
    # [xdot; qdot; vdot; ωdot]
end

# Use the StaticArrays methods since we know the size will always be small enough
@inline function dynamics!(model::RigidBody{D}, xdot, x, u, t=0) where {D}
    xdot .= dynamics(model, x, u, t)
end

# @inline wrenches(model::RigidBody, z::AbstractKnotPoint) = wrenches(model, state(z), control(z), time(z))
function wrenches(model::RigidBody, x, u, t)
    F = forces(model, x, u, t)
    M = moments(model, x, u, t)
    SA[F[1], F[2], F[3], M[1], M[2], M[3]]
end

@inline forces(model::RigidBody, x, u, t) = forces(model, x, u)
@inline moments(model::RigidBody, x, u, t) = moments(model, x, u)

@inline mass(::RigidBody) = throw(ErrorException("Not implemented"))
@inline inertia(::RigidBody)::SMatrix{3,3} = throw(ErrorException("Not implemented"))
@inline inertia_inv(model::RigidBody) = inv(inertia(model))
@inline forces(::RigidBody, x, u)::SVector{3} = throw(ErrorException("Not implemented"))
@inline moments(::RigidBody, x, u)::SVector{3} = throw(ErrorException("Not implemented"))
@inline velocity_frame(::RigidBody) = :world # :body or :world

function jacobian!(model::RigidBody{<:QuatRotation}, F, y, x, u, t)
    iF = SA[1, 2, 3]
    iM = SA[4, 5, 6]
    ir, iq, iv, iω, iu = gen_inds(model)

    # Extract the info from the state and model
    r, q, v, ω = parse_state(model, x)
    m = mass(model)
    J = inertia(model)
    Jinv = inertia_inv(model)

    ξ = wrenches(model, x, u, t)
    f = SA[ξ[1], ξ[2], ξ[3]]  # forces in world frame
    τ = SA[ξ[4], ξ[5], ξ[6]]  # torques in body frame

    # Calculate the Jacobian wrt the wrench and multiply the blocks according to the sparsity
    F .= 0
    Jw = uview(get_data(F), 8:13, :)
    wrench_jacobian!(Jw, model, z)
    js = wrench_sparsity(model)
    if velocity_frame(model) == :world
        tmp = I * 1 / m
    else
        tmp = 1 / m * RotMatrix(inv(q))
    end
    js[1, 1] && (Jw[iF, ir] .= tmp * Jw[iF, ir])
    js[1, 2] && (Jw[iF, iq] .= tmp * Jw[iF, iq])
    js[1, 3] && (Jw[iF, iv] .= tmp * Jw[iF, iv])
    js[1, 4] && (Jw[iF, iω] .= tmp * Jw[iF, iω])
    js[1, 5] && (Jw[iF, iu] .= tmp * Jw[iF, iu])
    js[2, 1] && (Jw[iM, ir] .= Jinv * Jw[iM, ir])
    js[2, 2] && (Jw[iM, iq] .= Jinv * Jw[iM, iq])
    js[2, 3] && (Jw[iM, iv] .= Jinv * Jw[iM, iv])
    js[2, 4] && (Jw[iM, iω] .= Jinv * Jw[iM, iω])
    js[2, 5] && (Jw[iM, iu] .= Jinv * Jw[iM, iu])

    # Add in the parts of the Jacobian that are not functions of the wrench
    F[iq, iq] .= 0.5 * Rotations.rmult(Rotations.pure_quaternion(ω))
    F[iq, iω] .= 0.5 * Rotations.lmult(q) * Rotations.hmat()
    F[iω, iω] .= Jinv * (skew(J * ω) - skew(ω) * J) + F[iω, iω]
    if velocity_frame(model) == :world
        F[1, 8] += 1
        F[2, 9] += 1
        F[3, 10] += 1
    else
        F[ir, iq] .+= Rotations.∇rotate(q, v)
        F[ir, iv] .+= RotMatrix(q)
        F[iv, iq] .+= 1 / m * Rotations.∇rotate(inv(q), f) * Rotations.tmat()
        F[iv, iv] .+= -Rotations.skew(ω)
        F[iv, iω] .+= Rotations.skew(v)
    end

    return F
end

function wrench_jacobian!(F, model::RigidBody, z)
    function w(x)
        wrenches(model, StaticKnotPoint(z, x))
    end
    ForwardDiff.jacobian!(F, w, z.z)
end

"""
    wrench_sparsity(model::RigidBody)

Specify the sparsity of the wrench Jacobian of `model` as a `js = SMatrix{2,5,Bool,10}`.
The elements of `js` correspond to the block elements of the wrench Jacobian:

```julia
[∂F/∂r ∂F/∂q ∂F/∂v ∂F/∂ω ∂F/∂u;
 ∂M/∂r ∂M/∂q ∂M/∂v ∂M/∂ω ∂M/∂u]
```

where `js[i,j] = false` if the corresponding partial derivative is always zero.

Note that this is only for performance improvement of continuous-time Jacobians of rigid bodies;
specifying the sparsity is completely optional.

# Example
For a fully-actuated satellite where `F = q*u[1:3]` and `M = u[4:6]`, the wrench sparsity
would be

```julia
SA[false true  false false true;
   false false false false true]
```
"""
wrench_sparsity(model::RigidBody) = @SMatrix ones(Bool, 2, 5)
