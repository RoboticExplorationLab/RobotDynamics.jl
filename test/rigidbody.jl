using RobotDynamics
using Rotations
using Test
using LinearAlgebra
using StaticArrays
using BenchmarkTools
using ForwardDiff

# Temporary fix for Rotations
function Rotations.∇rotate(q::UnitQuaternion, r::AbstractVector)
    Rotations.check_length(r, 3)
    rhat = UnitQuaternion(zero(eltype(r)), r[1], r[2], r[3], false)
    R = Rotations.rmult(q)
    2*Rotations.vmat()*Rotations.rmult(q)'*Rotations.rmult(rhat)
end

v = @SVector rand(3)
q = rand(UnitQuaternion)
ForwardDiff.jacobian(q->UnitQuaternion(q,false)*v, Rotations.params(q)) ≈ Rotations.∇rotate(q,v)

struct Body{R} <: RigidBody{R} end
RobotDynamics.control_dim(::Body) = 6

function RobotDynamics.wrenches(model::Body, x, u)
    q = orientation(model, x)
    F = q * SA[u[1], u[2], u[3]]  # forces in the body frame
    M = SA[u[4], u[5], u[6]]      # moments in the body frame
    SA[F[1], F[2], F[3], M[1], M[2], M[3]]
end

function RobotDynamics.wrench_jacobian!(F, model::Body, z::AbstractKnotPoint)
    inds = RobotDynamics.gen_inds(model)
    r,q,v,ω = RobotDynamics.parse_state(model, state(z))
    u = control(z)
    i3 = SA[1,2,3]
    f = u[i3]
    F[i3, inds.q] .= Rotations.∇rotate(q, f)
    F[i3, inds.u[i3]] .= RotMatrix(q)
    for i = 4:6
        F[i, inds.u[i]] = 1
    end
    return F
end

function testind(model)
    ir,iq,iv,iω,iu = gen_inds(model)
    @SVector ones(control_dim(model))
end


RobotDynamics.mass(::Body) = 2.0
RobotDynamics.inertia(::Body) = Diagonal(SA[2,3,1.])
RobotDynamics.wrench_sparsity(::RigidBody) = SA[false true  false false true;
                                                false false false false true]

model = Body{UnitQuaternion{Float64}}()
@test size(model) == (13,6)
x,u = rand(model)
q = orientation(model, x)
@test Rotations.params(q) == x[4:7]
z = KnotPoint(x,u,0.01)
@test length(x) == 13
@test length(u) == 6
@test norm(x[4:7]) ≈ 1

r,q,v,ω = RobotDynamics.parse_state(model, x)

# Test gen_inds
inds = RobotDynamics.gen_inds(model)
@test inds.q == 4:7
@test inds.u == 14:19
@test inds.r == 1:3
@test inds.v isa SVector{3,Int}

model2 = Body{MRP{Float64}}()
inds = RobotDynamics.gen_inds(model2)
@test inds.q == 4:6
@test inds.v == 7:9
@test inds.u == 13:18

ir,iq,iv,iω,iu = RobotDynamics.gen_inds(model)

using RobotDynamics: mass, inertia

# Test dynamics
RobotDynamics.velocity_frame(::Body) = :world
xdot = dynamics(model, x, u)
@test xdot ≈ dynamics(model, x, u, 1.0)
@test xdot ≈ dynamics(model, z)
xdot = RBState(xdot)
x_ = RBState(x)
@test position(xdot) ≈ linear_velocity(x_)
@test Rotations.params(orientation(xdot)) ≈ normalize(Rotations.kinematics(orientation(x_), angular_velocity(x_)))
ξ = RobotDynamics.wrenches(model, z)
F = ξ[SA[1,2,3]]
T = ξ[SA[4,5,6]]
@test linear_velocity(xdot) ≈ F/mass(model)
@test angular_velocity(xdot) ≈
	inertia(model) \ (T - angular_velocity(x_) × (inertia(model) * angular_velocity(x_)))

# Test body-frame velocity
RobotDynamics.velocity_frame(::Body) = :body
xdot = dynamics(model, x, u)
@test xdot ≈ dynamics(model, x, u, 1.0)
@test xdot ≈ dynamics(model, z)
xdot = RBState(xdot)
x_ = RBState(x)
q = orientation(x_)
@test position(xdot) ≈ q * linear_velocity(x_)
@test Rotations.params(orientation(xdot)) ≈ normalize(Rotations.kinematics(orientation(x_), angular_velocity(x_)))
@test linear_velocity(xdot) ≈ q \ (F / mass(model)) - angular_velocity(x_) × linear_velocity(x_)
@test angular_velocity(xdot) ≈
	inertia(model) \ (T - angular_velocity(x_) × (inertia(model) * angular_velocity(x_)))
