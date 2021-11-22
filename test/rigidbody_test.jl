using RobotDynamics
using Rotations
using Test
using LinearAlgebra
using StaticArrays
using BenchmarkTools
using ForwardDiff

using RobotDynamics: mass, inertia

# Temporary fix for Rotations
# function Rotations.∇rotate(q::UnitQuaternion, r::AbstractVector)
#     Rotations.check_length(r, 3)
#     rhat = UnitQuaternion(zero(eltype(r)), r[1], r[2], r[3], false)
#     R = Rotations.rmult(q)
#     2*Rotations.vmat()*Rotations.rmult(q)'*Rotations.rmult(rhat)
# end

v = @SVector rand(3)
q = rand(UnitQuaternion)
ForwardDiff.jacobian(q->UnitQuaternion(q,false)*v, Rotations.params(q)) ≈ Rotations.∇rotate(q,v)

RD.@autodiff struct Body{R} <: RobotDynamics.RigidBody{R} end
RobotDynamics.control_dim(::Body) = 6

function RobotDynamics.wrenches(model::Body, x::StaticVector, u::StaticVector, t)
    q = orientation(model, x)
    F = q * SA[u[1], u[2], u[3]]  # forces in the body frame
    M = SA[u[4], u[5], u[6]]      # moments in the body frame
    SA[F[1], F[2], F[3], M[1], M[2], M[3]]
end

function RobotDynamics.wrench_jacobian!(F, model::Body, z::RD.AbstractKnotPoint)
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
RobotDynamics.wrench_sparsity(::Body) = SA[false true  false false true;
                                           false false false false true]

#---
model = Body{UnitQuaternion{Float64}}()
@test RD.dims(model) == (13,6,13)
x,u = rand(model)
q = orientation(model, x)
@test Rotations.params(q) == x[4:7]
@test norm(q) ≈ 1
z = RD.KnotPoint(x,u,0.0,0.01)
@test length(x) == 13
@test length(u) == 6
@test norm(x[4:7]) ≈ 1
@test RobotDynamics.LieState(model) === RobotDynamics.LieState(UnitQuaternion{Float64},(3,6))
@test RobotDynamics.rotation_type(model) == UnitQuaternion{Float64}

# Test initializers
x0,u0 = zeros(model)
@test RBState(model, x0) ≈ zero(RBState)
@test u0 ≈ zeros(RD.control_dim(model))
@test x0[4:7] ≈ [1,0,0,0]

# Test diferent rotations
for R in [UnitQuaternion{Float64}, MRP{Float64}, RodriguesParam{Float64}]
    local model = Body{R}()
	@test RD.state_dim(model) == 9 + Rotations.params(R)
	RobotDynamics.rotation_type(model) == R
	local x0,u0 = zeros(model)
	@test RBState(model, x0) ≈ zero(RBState)
end

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

# Test state building methods
r,q,v,ω = RobotDynamics.parse_state(model, x)
ir,iq,iv,iω,iu = RobotDynamics.gen_inds(model)
@test r == x[ir]
@test Rotations.params(q) == x[iq]
@test v == x[iv]
@test ω == x[iω]

x_ = RBState(r, q, v, ω)
@test RobotDynamics.build_state(model, r, q, v, ω) ≈ x
@test RobotDynamics.build_state(model, r, MRP(q), v, ω) ≈ x
@test RobotDynamics.build_state(model, Vector(r), MRP(q), v, ω) ≈ x
@test RobotDynamics.build_state(model, x_) ≈ x
@test RobotDynamics.build_state(model, SVector(x_)) ≈ x
@test RobotDynamics.build_state(model, Vector(x_)) ≈ x

model2 = Body{MRP{Float64}}()
g = Rotations.params(MRP(q))
x2 = [r; g; v; ω]
@test RobotDynamics.build_state(model2, r, q, v, ω) ≈ x2
@test RobotDynamics.build_state(model2, r, MRP(q), v, ω) ≈ x2
@test RobotDynamics.build_state(model2, r, g, v, ω) ≈ x2

@test RobotDynamics.fill_state(model, 1, 2, 3, 4) isa SVector{13,Int}
@test RobotDynamics.fill_state(model2, 1, 2, 3, 4) isa SVector{12,Int}
@test RobotDynamics.fill_state(model, 1, 2, 3, 4.) isa SVector{13,Float64}
x_ = RobotDynamics.fill_state(model, 1, 2, 3, 4)
@test position(model, x_) ≈ fill(1,3)
@test Rotations.params(orientation(model, x_)) ≈ fill(2,4)
@test linear_velocity(model, x_) ≈ fill(3,3)
@test angular_velocity(model, x_) ≈ fill(4,3)
@test RobotDynamics.fill_state(model2, 1, 2, 3, 4.)[4:6] ≈ fill(2,3)


# Test dynamics
RobotDynamics.velocity_frame(::Body) = :world
xdot = RD.dynamics(model, x, u)
@test xdot ≈ RD.dynamics(model, x, u, 1.0)
@test xdot ≈ RD.dynamics(model, z)
xdot = RBState(xdot)
x_ = RBState(x)
@test position(xdot) ≈ linear_velocity(x_)
@test Rotations.params(orientation(xdot)) ≈ Rotations.kinematics(orientation(x_), angular_velocity(x_))
ξ = RobotDynamics.wrenches(model, z)
F = ξ[SA[1,2,3]]
T = ξ[SA[4,5,6]]
@test linear_velocity(xdot) ≈ F/mass(model)
@test angular_velocity(xdot) ≈
	inertia(model) \ (T - angular_velocity(x_) × (inertia(model) * angular_velocity(x_)))

# Test body-frame velocity
RobotDynamics.velocity_frame(::Body) = :body
xdot = RD.dynamics(model, x, u)
@test xdot ≈ RD.dynamics(model, x, u, 1.0)
@test xdot ≈ RD.dynamics(model, z)
xdot = RBState(xdot)
x_ = RBState(x)
q = orientation(x_)
@test position(xdot) ≈ q * linear_velocity(x_)
@test Rotations.params(orientation(xdot)) ≈ Rotations.kinematics(orientation(x_), angular_velocity(x_))
@test linear_velocity(xdot) ≈ q \ (F / mass(model)) - angular_velocity(x_) × linear_velocity(x_)
@test angular_velocity(xdot) ≈
	inertia(model) \ (T - angular_velocity(x_) × (inertia(model) * angular_velocity(x_)))



# Test flipquat
x = rand(model)[1]
x_ = RobotDynamics.flipquat(model, x)
@test orientation(model, x) ≈ orientation(model, x_)
@test Rotations.params(orientation(model, x)) ≈ -Rotations.params(orientation(model, x_))

# Test RBState methods
x_ = RBState(model, x)
@test position(model, x) ≈ position(x_)
@test orientation(model, x) ≈ orientation(x_)
@test linear_velocity(model, x) ≈ linear_velocity(x_)
@test angular_velocity(model, x) ≈ angular_velocity(x_)
@test RBState(model, x_) isa RBState

# Not Implemented Body
struct FakeBody{R} <: RD.RigidBody{R} end
model = FakeBody{MRP{Float64}}()
@test_throws ErrorException("Not implemented") mass(model)
@test_throws ErrorException("Not implemented") inertia(model)
@test_throws ErrorException("Not implemented") RobotDynamics.forces(model, x, u)
@test_throws ErrorException("Not implemented") RobotDynamics.moments(model, x, u)
@test RobotDynamics.velocity_frame(model) == :world
