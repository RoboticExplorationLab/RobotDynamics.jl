using RobotDynamics
using LinearAlgebra
using StaticArrays
using Rotations
using BenchmarkTools
using Test

struct SatellitePair{R,T} <: LieGroupModel
    J1::SMatrix{3,3,T,9}   # inertia of satellite 1
    J2::SMatrix{3,3,T,9}   # inertia of satellite 2
end

function RobotDynamics.dynamics(model::SatellitePair, x, u)
    vs = RobotDynamics.vec_states(model, x)
    qs = RobotDynamics.rot_states(model, x)
    ω1 = vs[2]
    ω2 = vs[3]
    q1 = qs[1]
    q2 = qs[2]

    J1, J2 = model.J1, model.J2
    u1 = u[SA[1,2,3]]
    u2 = u[SA[4,5,6]]
    ω1dot = J1\(u1 - ω1 × (J1 * ω1))
    ω2dot = J2\(u2 - ω2 × (J2 * ω2))
    q1dot = Rotations.kinematics(q1, ω1)
    q2dot = Rotations.kinematics(q2, ω2)
    SA[
        q1dot[1], q1dot[2], q1dot[3], q1dot[4],
        ω1dot[1], ω1dot[2], ω1dot[3],
        q2dot[1], q2dot[2], q2dot[3], q2dot[4],
        ω2dot[1], ω2dot[2], ω2dot[3],
    ]
end

RobotDynamics.control_dim(::SatellitePair) = 6

RobotDynamics.LieState(::SatellitePair{R}) where R = RobotDynamics.LieState(R, (0,3,3))

J1 = SMatrix{3,3}(Diagonal(fill(1.0, 3)))
J2 = SMatrix{3,3}(Diagonal(fill(2.0, 3)))
model = SatellitePair{UnitQuaternion{Float64}, Float64}(J1, J2)
x,u = rand(model)
@test norm(x[1:4]) ≈ 1
@test norm(x[8:11]) ≈ 1
s = RobotDynamics.LieState(model)
@test length(s) == 14

ω1 = x[5:7]
ω2 = x[12:14]
q1 = UnitQuaternion(x[1:4],false)
q2 = UnitQuaternion(x[8:11],false)
@test all(RobotDynamics.vec_states(s, x)[2:3] .≈ (ω1, ω2))
@test all(RobotDynamics.rot_states(s, x) .≈ (q1,q2))


model = SatellitePair{MRP{Float64}, Float64}(J1, J2)
x,u = rand(model)
@test length(x) == 12
s = RobotDynamics.LieState(model)
@test length(s) == 12

ω1 = x[4:6]
ω2 = x[10:12]
q1 = MRP(x[1:3]...)
q2 = MRP(x[7:9]...)
@test all(RobotDynamics.vec_states(s, x)[2:3] .≈ (ω1, ω2))
@test all(RobotDynamics.rot_states(s, x) .≈ (q1,q2))
