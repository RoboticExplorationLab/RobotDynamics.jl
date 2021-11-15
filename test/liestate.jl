using Test
using StaticArrays
using Rotations
using LinearAlgebra
using BenchmarkTools
using RobotDynamics

import RobotDynamics: LieState
const RD = RobotDynamics

function error_state_quat(x::AbstractVector, x0::AbstractVector)
    dq1 = UnitQuaternion(x[4],x[5],x[6],x[7]) ⊖ UnitQuaternion(x0[4],x0[5],x0[6],x0[7])
    dq2 = UnitQuaternion(x[10],x[11],x[12],x[13]) ⊖ UnitQuaternion(x0[10],x0[11],x0[12],x0[13])
    dx = x - x0
    SA[dx[1], dx[2], dx[3], dq1[1], dq1[2], dq1[3], dx[8], dx[9], dq2[1], dq2[2], dq2[3], dx[14], dx[15], dx[16]]
end

function error_state_rp(x::AbstractVector, x0::AbstractVector)
    dq1 = RodriguesParam(x[4],x[5],x[6]) ⊖ RodriguesParam(x0[4],x0[5],x0[6])
    dq2 = RodriguesParam(x[9],x[10],x[11]) ⊖ RodriguesParam(x0[9],x0[10],x0[11])
    dx = x - x0
    SA[dx[1], dx[2], dx[3], dq1[1], dq1[2], dq1[3],
       dx[7], dx[8],        dq2[1], dq2[2], dq2[3], dx[12], dx[13], dx[14]]
end

# Test params
@test Rotations.params(UnitQuaternion) == 4
@test Rotations.params(RodriguesParam) == 3
@test Rotations.params(MRP) == 3
@test Rotations.params(RotMatrix3) == 9
@test Rotations.params(RotMatrix2) == 4

# Test index functions
R = UnitQuaternion{Float64}
P = (3,2,3)
s = LieState{R,P}()
@test length(s) == 16
@test RD.errstate_dim(s) == 14
@test RobotDynamics.vec_inds(R,P,1) == 1:3   == RobotDynamics.inds(R,P,1)
@test RobotDynamics.rot_inds(R,P,1) == 4:7   == RobotDynamics.inds(R,P,2)
@test RobotDynamics.vec_inds(R,P,2) == 8:9   == RobotDynamics.inds(R,P,3)
@test RobotDynamics.rot_inds(R,P,2) == 10:13 == RobotDynamics.inds(R,P,4)
@test RobotDynamics.vec_inds(R,P,3) == 14:16 == RobotDynamics.inds(R,P,5)
@test RobotDynamics.QuatState(16, SA[4,10]) === s
@test RobotDynamics.QuatState(16, @MVector [4,10]) === s
@test RobotDynamics.QuatState(16, (4,10)) === s
@test RobotDynamics.num_rotations(s) == 2
@test length(typeof(s)) == length(s)

n = length(s)
x  = @MVector rand(n)
x0 = @MVector rand(n)
@test RobotDynamics.state_diff(s, x, x0) ≈ error_state_quat(x, x0)

# Test with RP
R = RodriguesParam{Float64}
P = (3,2,3)
s = LieState{R,P}()
@test LieState(R,P) === s
@test LieState(R,3,2,3) === s
@test length(s) == 14
@test RD.errstate_dim(s) == 14
@test RobotDynamics.vec_inds(R,P,1) == 1:3   == RobotDynamics.inds(R,P,1)
@test RobotDynamics.rot_inds(R,P,1) == 4:6   == RobotDynamics.inds(R,P,2)
@test RobotDynamics.vec_inds(R,P,2) == 7:8   == RobotDynamics.inds(R,P,3)
@test RobotDynamics.rot_inds(R,P,2) == 9:11  == RobotDynamics.inds(R,P,4)
@test RobotDynamics.vec_inds(R,P,3) == 12:14 == RobotDynamics.inds(R,P,5)
@test RobotDynamics.state_diff(s, x, x0) ≈ error_state_rp(x, x0)

# @btime error_state($s, $x, $x0)
# @btime error_state_rp($x, $x0)

# state diff jacobian
R = UnitQuaternion{Float64}
P = (3,2,3)
s = LieState{R,P}()
n = length(s)
n̄ = RD.errstate_dim(s)
G = zeros(n,n̄)
RobotDynamics.state_diff_jacobian!(G, s, x)

q1 = UnitQuaternion(x[4],x[5],x[6],x[7])
q2 = UnitQuaternion(x[10],x[11],x[12],x[13])
G1 = Rotations.∇differential(q1)
G2 = Rotations.∇differential(q2)
G0 = cat(I(3),G1,I(2),G2,I(3),dims=(1,2))
@test G0 ≈ G

# state diff jacobian jacobian
∇G = SizedMatrix{n̄,n̄}(zeros(n̄,n̄))
dx = @SVector rand(n)

dq1 = SVector(dx[4],dx[5],dx[6],dx[7])
dq2 = SVector(dx[10],dx[11],dx[12],dx[13])
RobotDynamics.∇²differential!(∇G, s, x, dx)
∇G1 = Rotations.∇²differential(q1, dq1)
∇G2 = Rotations.∇²differential(q2, dq2)
∇G0 = cat(zeros(3,3), ∇G1, zeros(2,2), ∇G2, zeros(3,3), dims=(1,2))
@test Matrix(∇G0) ≈ Matrix(∇G)

struct LieBody <: RD.LieGroupModel end
RobotDynamics.control_dim(::LieBody) = 4
RobotDynamics.LieState(::LieBody) = LieState(UnitQuaternion{Float64},(3,2,3))

model = LieBody()
@test s === LieState(model)
G .= 0
∇G .= 0
RobotDynamics.state_diff_jacobian!(G, model, x)
RobotDynamics.∇²differential!(∇G, model, x, dx)
@test G0 ≈ G
@test ∇G0 ≈ ∇G
