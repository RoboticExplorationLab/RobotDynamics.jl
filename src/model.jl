export
    AbstractModel,
    InfeasibleModel,
    RigidBody,
    dynamics,
    discrete_dynamics,
    jacobian!,
    discrete_jacobian!,
	orientation,
	state_dim,
	control_dim,
	state_diff_size

export
    QuadratureRule,
	RK2,
    RK3,
	RK4,
    HermiteSimpson


""" $(TYPEDEF)
Abstraction of a model of a dynamical system of the form ẋ = f(x,u), where x is the n-dimensional state vector
and u is the m-dimensional control vector.

Any inherited type must define the following interface:
ẋ = dynamics(model, x, u)
n,m = size(model)
"""
abstract type AbstractModel end


""" $(TYPEDEF)
Abstraction of a dynamical system with free-body dynamics, with a 12 or 13-dimensional state
vector: `[p; q; v; ω]`
where `p` is the 3D position, `q` is the 3 or 4-dimension attitude representation, `v` is the
3D linear velocity, and `ω` is the 3D angular velocity.

# Interface
Any single-body system can leverage the `RigidBody` type by inheriting from it and defining the
following interface:
```julia
forces(::MyRigidBody, x, u)  # return the forces in the world frame
moments(::MyRigidBody, x, u) # return the moments in the body frame
inertia(::MyRigidBody, x, u) # return the 3x3 inertia matrix
inertia_inv(::MyRigidBody, x, u)  # return the 3x3 inverse of the inertia matrix
mass_matrix(::MyRigidBody, x, u)  # return the 3x3 mass matrix
```

# Rotation Parameterization
A `RigidBody` model must specify the rotational representation being used. Any of the following
can be used:
* [`UnitQuaternion`](@ref): Unit Quaternion. Note that this representation needs to be further parameterized.
* [`MRP`](@ref): Modified Rodrigues Parameters
* [`RPY`](@ref): Roll-Pitch-Yaw Euler angles
"""
abstract type RigidBody{R<:Rotation} <: AbstractModel end

"Integration rule for approximating the continuous integrals for the equations of motion"
abstract type QuadratureRule end

"Integration rules of the form x′ = f(x,u), where x′ is the next state"
abstract type Implicit <: QuadratureRule end

"Integration rules of the form x′ = f(x,u,x′,u′), where x′,u′ are the states and controls at the next time step."
abstract type Explicit <: QuadratureRule end

"Fourth-order Runge-Kutta method with zero-order-old on the controls"
abstract type RK4 <: Implicit end

"Second-order Runge-Kutta method with zero-order-old on the controls"
abstract type RK3 <: Implicit end

"Second-order Runge-Kutta method with zero-order-old on the controls"
abstract type RK2 <: Implicit end

"Third-order Runge-Kutta method with first-order-hold on the controls"
abstract type HermiteSimpson <: Explicit end

"Default quadrature rule"
const DEFAULT_Q = RK3


#=
Convenient methods for creating state and control vectors directly from the model
=#
for method in [:rand, :zeros, :ones]
    @eval begin
        function Base.$(method)(model::AbstractModel)
            n,m = size(model)
            x = @SVector $(method)(n)
            u = @SVector $(method)(m)
            return x, u
        end
        function Base.$(method)(::Type{T}, model::AbstractModel) where T
            n,m = size(model)
            x = @SVector $(method)(T,n)
            u = @SVector $(method)(T,m)
            return x,u
        end
    end
end
function Base.fill(model::AbstractModel, val)
    n,m = size(model)
    x = @SVector fill(val,n)
    u = @SVector fill(val,m)
    return x, u
end

@inline control_dim(::AbstractModel) = throw(ErrorException("control_dim not implemented"))
@inline state_dim(::AbstractModel) = throw(ErrorException("state_dim not implemented"))

"""Default size method for model (assumes model has fields n and m)"""
@inline Base.size(model::AbstractModel) = model.n, model.m

############################################################################################
#                               CONTINUOUS TIME METHODS                                    #
############################################################################################
"""```
ẋ = dynamics(model, z::KnotPoint)
```
Compute the continuous dynamics of a dynamical system given a KnotPoint"""
@inline dynamics(model::AbstractModel, z::KnotPoint) = dynamics(model, state(z), control(z), z.t)

# Default to not passing in t
@inline dynamics(model::AbstractModel, x, u, t) = dynamics(model, x, u)

"""```
∇f = jacobian!(∇c, model, z::KnotPoint)
∇f = jacobian!(∇c, model, z::SVector)
```
Compute the Jacobian of the continuous-time dynamics using ForwardDiff. The input can be either
a static vector of the concatenated state and control, or a KnotPoint. They must be concatenated
to avoid unnecessary memory allocations.
"""
@inline jacobian!(∇c::SizedMatrix, model::AbstractModel, z::AbstractKnotPoint) =
	jacobian!(∇c.data, model, z)
function jacobian!(∇c::Matrix, model::AbstractModel, z::AbstractKnotPoint)
    ix, iu = z._x, z._u
    f_aug(z) = dynamics(model, z[ix], z[iu])
    s = z.z
	ForwardDiff.jacobian!(∇c, f_aug, s)
end

# # QUESTION: is this one needed?
# function jacobian!(∇c, model::AbstractModel, z::SVector)
#     n,m = size(model)
#     ix,iu = 1:n, n .+ (1:m)
#     f_aug(z) = dynamics(model, view(z,ix), view(z,iu))
#     ForwardDiff.jacobian!(∇c, f_aug, z)
# end

############################################################################################
#                          IMPLICIT DISCRETE TIME METHODS                                  #
############################################################################################

# Set default integrator
@inline discrete_dynamics(model::AbstractModel, z::KnotPoint) =
    discrete_dynamics(DEFAULT_Q, model, z)

""" Compute the discretized dynamics of `model` using implicit integration scheme `Q<:QuadratureRule`.

Methods:
```
x′ = discrete_dynamics(model, model, z)  # uses $(DEFAULT_Q) as the default integration scheme
x′ = discrete_dynamics(Q, model, x, u, t, dt)
x′ = discrete_dynamics(Q, model, z::KnotPoint)
```

The default integration scheme is stored in `TrajectoryOptimization.DEFAULT_Q`
"""
@inline discrete_dynamics(::Type{Q}, model::AbstractModel, z::KnotPoint) where Q<:Implicit =
    discrete_dynamics(Q, model, state(z), control(z), z.t, z.dt)


"Propagate the dynamics forward, storing the result in the next knot point"
function propagate_dynamics(::Type{Q}, model::AbstractModel, z_::KnotPoint, z::KnotPoint) where Q<:Implicit
    x_next = discrete_dynamics(Q, model, z)
    set_state!(z_, x_next)
end

""" Compute the discrete dynamics Jacobian of `model` using implicit integration scheme `Q<:QuadratureRule`

Methods:
```
∇f = discrete_dynamics(model, z::KnotPoint)  # uses $(DEFAULT_Q) as the default integration scheme
∇f = discrete_jacobian(Q, model, z::KnotPoint)
∇f = discrete_jacobian(Q, model, s::SVector{NM1}, t, ix::SVector{N}, iu::SVector{M})
```
where `s = [x; u; dt]`, `t` is the time, and `ix` and `iu` are the indices to extract the state and controls.
"""
function discrete_jacobian!(::Type{Q}, ∇f, model::AbstractModel,
		z::AbstractKnotPoint{T,N,M,NM}) where {T,N,M,NM,Q<:Implicit}
    ix,iu,idt = z._x, z._u, NM+1
    t = z.t
    fd_aug(s) = discrete_dynamics(Q, model, s[ix], s[iu], t, s[idt])
    s = [z.z; @SVector [z.dt]]
    ForwardDiff.jacobian!(∇f, fd_aug, s)
end

function dynamics_expansion!(D::Vector{<:DynamicsExpansion}, model::AbstractModel,
		Z::Traj)
	for k in eachindex(D)
		discrete_jacobian!(RK3, D[k].∇f, model, Z[k])
		D[k].tmpA .= D[k].A_  # avoids allocations later
		D[k].tmpB .= D[k].B_
	end
end

function error_expansion!(D::DynamicsExpansion,G1,G2)
    mul!(D.tmp, D.tmpA, G1)
    mul!(D.A, Transpose(G2), D.tmp)
    mul!(D.B, Transpose(G2), D.tmpB)
end

@inline error_expansion(D::DynamicsExpansion, model::RigidBody) = D.A, D.B
@inline error_expansion(D::DynamicsExpansion, model::AbstractModel) = D.tmpA, D.tmpB
@inline DynamicsExpansion(model::AbstractModel) = DynamicsExpansion{Float64}(model)
@inline function DynamicsExpansion{T}(model::AbstractModel) where T
	n,m = size(model)
	n̄ = state_diff_size(model)
	DynamicsExpansion{T}(n,n̄,m)
end

############################################################################################
#                               STATE DIFFERENTIALS                                        #
############################################################################################

state_diff(model::AbstractModel, x, x0) = x - x0
@inline state_diff_jacobian(model::AbstractModel, x::SVector{N,T}) where {N,T} = I
@inline state_diff_size(model::AbstractModel) = size(model)[1]

@inline state_diff_jacobian!(G, model::AbstractModel, Z::Traj) = nothing

function ∇²differential!(G, model::AbstractModel, x::SVector, dx::AbstractVector)
	G .= ∇²differential(model, x, dx)
end
