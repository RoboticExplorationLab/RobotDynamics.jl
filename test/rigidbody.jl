using RobotDynamics
using Rotations
using Test
using LinearAlgebra
using StaticArrays
using BenchmarkTools

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

@btime dynamics($model, $x, $u)
F = zeros(13,19)
jacobian!(F, model, z)

F2 = zero(F)
RobotDynamics.rb_jacobian!(F2, model, z)
@test F2 ≈ F

D = RobotDynamics.DynamicsJacobian(13,6)
jacobian!(D, model, z)
@test D ≈ F

# @btime RobotDynamics.rb_jacobian!($F2, $model, $z)
# @btime jacobian!($F, $model, $z)
# @btime RobotDynamics.rb_jacobian!($D, $model, $z)


discrete_jacobian!(RK3, F, model, z)
tmp = [RobotDynamics.DynamicsJacobian(13,6) for k = 1:3]
jacobian!(RK3, D, model, z, tmp)
F ≈ D

@btime discrete_jacobian!($RK3, $F, $model, $z)
@btime jacobian!($RK3, $D, $model, $z, $tmp)
