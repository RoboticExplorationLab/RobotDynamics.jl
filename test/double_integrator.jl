using RobotDynamics
const RD = RobotDynamics

using StaticArrays
using Test
using BenchmarkTools

struct DoubleIntegrator{T} <: ContinuousLTI
    A::SMatrix{2,2,T,4}
    B::SMatrix{2,1,T,2}
end

function DoubleIntegrator()
    A = [   0.0 1.0;
            0.0 0.0 ]
    B = [   0.0; 
            1.0 ]
    return DoubleIntegrator{Float64}(A, B)
end

RD.state_dim(::DoubleIntegrator) = 2
RD.control_dim(::DoubleIntegrator) = 1
RD.get_A(model::DoubleIntegrator) = model.A
RD.get_B(model::DoubleIntegrator) = model.B

model = DoubleIntegrator()

@test RD.is_affine(model) == Val(false)
@test RD.is_time_varying(model) == false

@test RD.get_A(model, 10) === model.A
@test RD.get_B(model, 10) === model.B

dt = 0.01
t = 0.1

x, u = rand(model)

ẋ = dynamics(model, x, u, t)
@test model.A * x + model.B * u ≈ ẋ atol=1e-12


########################################################################################
struct DoubleIntegratorAffine{T} <: ContinuousLTI
    A::SMatrix{2,2,T,4}
    B::SMatrix{2,1,T,2}
    d::SVector{2,T}
end

function DoubleIntegratorAffine()
    A = [   0.0 1.0;
            0.0 0.0 ]
    B = [   0.0; 
            1.0 ]
    d = [   0.0;
            -9.81]
    return DoubleIntegratorAffine{Float64}(A, B, d)
end

RD.is_affine(::DoubleIntegratorAffine) = Val(true)
RD.state_dim(::DoubleIntegratorAffine) = 2
RD.control_dim(::DoubleIntegratorAffine) = 1
RD.get_A(model::DoubleIntegratorAffine) = model.A
RD.get_B(model::DoubleIntegratorAffine) = model.B
RD.get_d(model::DoubleIntegratorAffine) = model.d

affine_model = DoubleIntegratorAffine()

@test RD.get_A(affine_model, 10) === affine_model.A
@test RD.get_B(affine_model, 10) === affine_model.B
@test RD.get_d(affine_model, 10) === affine_model.d

ẋ = dynamics(affine_model, x, u, t)
@test affine_model.A * x + affine_model.B * u + affine_model.d ≈ ẋ atol=1e-12