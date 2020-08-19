using RobotDynamics
using StaticArrays
using LinearAlgebra
using BenchmarkTools
using Test

## Setup
include("random_linear.jl")

N = 11
n,m = 15,10
dt = 0.1
times = range(0,step=dt,length=N)
t = times[3] 
Ad,Bd = gencontrollable(n,m)
@test isstabled(Ad)
x = @SVector rand(n) 
u = @SVector rand(m) 
z = KnotPoint(x,u,dt)


## Discrete Time Invariant
struct DLTI{TA,TB} <: DiscreteLTI
    A::TA
    B::TB
end
RobotDynamics.state_dim(::DLTI) = 15 
RobotDynamics.control_dim(::DLTI) = 10

model = DLTI(SMatrix{n,n}(Ad),SMatrix{n,m}(Bd))
@test RobotDynamics.is_affine(model) == Val(false)
@test RobotDynamics.is_time_varying(model) == false

@test_throws ErrorException discrete_dynamics(DiscreteLinearQuadrature, model, x, u, t, dt)
RobotDynamics.get_A(model::DLTI) = model.A
RobotDynamics.get_B(model::DLTI) = model.B
x2 = discrete_dynamics(DiscreteLinearQuadrature, model, x, u, t, dt)
@test x2 ≈ Ad*x + Bd*u
@test (@ballocated discrete_dynamics(DiscreteLinearQuadrature, $model, $x, $u, $t, $dt) samples=2 evals=2) == 0

F = RobotDynamics.DynamicsJacobian(model)
discrete_jacobian!(DiscreteLinearQuadrature, F, model, z)
@test F ≈ [Ad Bd]


struct DLTV{TA,TB} <: DiscreteLTV
    A::Vector{TA}
    B::Vector{TB}
    times::Vector{T}
end
RobotDynamics.state_dim(::DLTV) = 15 
RobotDynamics.control_dim(::DLTV) = 10
RobotDynamics.get_A(model::DLTV, k::Int) = model.A[k]
RobotDynamics.get_B(model::DLTV, k::Int) = model.B[k]

N = 11
systems = [gencontrollable(n,m) for k = 1:N]
A = [SMatrix{n,n}(s[1]) for s in systems]
B = [SMatrix{n,m}(s[2]) for s in systems]

model = DLTV(A,B,times)

@test RobotDynamics.is_affine(model) == Val(false)
@test RobotDynamics.is_time_varying(model) == true 

@test_throws ErrorException discrete_dynamics(DiscreteLinearQuadrature, model, x, u, t, dt)
RobotDynamics.get_times(::DLTV) = 
x2 = discrete_dynamics(DiscreteLinearQuadrature, model, x, u, t, dt)
@test x2 ≈ A[3]*x + B[3]*u
@test !(x2 ≈ A[1]*x + B[1]*u)

@ballocated discrete_dynamics(DiscreteLinearQuadrature, $model, $x, $u, $t, $dt)
F = RobotDynamics.DynamicsJacobian(model)
discrete_jacobian!(DiscreteLinearQuadrature, F, model, z)
@test F ≈ [Ad Bd]
