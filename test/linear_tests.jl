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
z = KnotPoint(x,u,dt,t)


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

@test_throws ErrorException discrete_dynamics(DiscreteSystemQuadrature, model, x, u, t, dt)
RobotDynamics.get_A(model::DLTI) = model.A
RobotDynamics.get_B(model::DLTI) = model.B
x2 = discrete_dynamics(DiscreteSystemQuadrature, model, x, u, t, dt)
@test x2 ≈ Ad*x + Bd*u
@test (@ballocated discrete_dynamics(DiscreteSystemQuadrature, $model, $x, $u, $t, $dt) samples=2 evals=2) == 0

F = RobotDynamics.DynamicsJacobian(model)
discrete_jacobian!(DiscreteSystemQuadrature, F, model, z)
@test F ≈ [Ad Bd]


## DLTV 
struct DLTV{TA,TB} <: DiscreteLTV
    A::Vector{TA}
    B::Vector{TB}
    times::Vector{Float64}
end
RobotDynamics.state_dim(::DLTV) = 15 
RobotDynamics.control_dim(::DLTV) = 10
RobotDynamics.get_A(model::DLTV, k::Int) = model.A[k]
RobotDynamics.get_B(model::DLTV, k::Int) = model.B[k]

N = 11
systems = [gencontrollable(n,m) for k = 1:N]
A = [SMatrix{n,n}(s[1]) for s in systems]
B = [SMatrix{n,m}(s[2]) for s in systems]

model = DLTV(A,B,collect(times))

@test RobotDynamics.is_affine(model) == Val(false)
@test RobotDynamics.is_time_varying(model) == true 

@test_throws ErrorException discrete_dynamics(DiscreteSystemQuadrature, model, x, u, t, dt)
RobotDynamics.get_times(model::DLTV) = model.times
x2 = discrete_dynamics(DiscreteSystemQuadrature, model, x, u, t, dt)
@test x2 ≈ A[3]*x + B[3]*u
@test !(x2 ≈ A[1]*x + B[1]*u)
@test (@ballocated discrete_dynamics(DiscreteSystemQuadrature, $model, $x, $u, $t, $dt)) == 0

F = RobotDynamics.DynamicsJacobian(model)
discrete_jacobian!(DiscreteSystemQuadrature, F, model, z)
@test F ≈ [A[3] B[3]]
@test !(F ≈ [A[1] B[1]])

## CLTI
struct CLTI{TA,TB} <: ContinuousLTI 
    A::TA
    B::TB
end
RobotDynamics.state_dim(::CLTI) = 15 
RobotDynamics.control_dim(::CLTI) = 10

model = CLTI(SMatrix{n,n}(Ad),SMatrix{n,m}(Bd))
@test_throws ErrorException dynamics(model, x, u, t)
RobotDynamics.get_A(model::CLTI) = model.A
RobotDynamics.get_B(model::CLTI) = model.B
xdot = dynamics(model, x, u, t)
@test xdot ≈ Ad*x + Bd*u
@test (@ballocated dynamics($model, $x, $u, $t)) == 0

jacobian!(F, model, z)
@test F ≈ [Ad Bd]

x2 = discrete_dynamics(RK3, model, z)
begin
    local k1_ = (Ad*x + Bd*u)*dt
    local k2_ = (Ad*(x + k1_/2) + Bd*u)*dt
    local k3_ = (Ad*(x - k1_ + 2k2_) + Bd*u)*dt
    @test x2 ≈ x + (k1_+4k2_+k3_)/6
end

discrete_jacobian!(RK3, F, model, z)
@test F.A ≈ I(n) + (Ad*dt + 4*Ad*(I + Ad*dt/2)*dt + Ad*(I - Ad*dt + 2*Ad*(I + Ad*dt/2)*dt)*dt)/6
@test F.B ≈ (Bd*dt + 4*(Ad'Bd*dt/2 + Bd)*dt + (Ad'*(-Bd*dt + 2*(Ad'Bd*dt/2 + Bd)*dt) + Bd)*dt)/6 
@test (@ballocated discrete_jacobian!(RK3, $F, $model, $z)) == 0

## CLTV
struct CLTV{TA,TB} <: ContinuousLTV 
    A::Vector{TA}
    B::Vector{TB}
    times::Vector{Float64}
end
RobotDynamics.state_dim(::CLTV) = 15 
RobotDynamics.control_dim(::CLTV) = 10
RobotDynamics.get_A(model::CLTV, k::Int) = model.A[k]
RobotDynamics.get_B(model::CLTV, k::Int) = model.B[k]

model = CLTV(A,B,collect(times))
@test is_time_varying(model)
@test is_affine(model) == Val(false)

@test_throws ErrorException dynamics(model, x, u, t)
RobotDynamics.get_times(model::CLTV) = model.times
xdot = dynamics(model, x, u, t)
@test xdot ≈ dynamics(model, z)
@test xdot ≈ A[3]*x + B[3]*u

jacobian!(F, model, z)
@test F ≈ [A[3] B[3]]

x2 = discrete_dynamics(RK2, model, z)
A_,B_ = A[3], B[3]
begin
    local k1_ = (A_*x + B_*u)*dt
    local k2_ = (A_*(x + k1_/2) + B_*u)*dt
    @test x2 ≈ x + k2_ 
end

discrete_jacobian!(RK2, F, model, z)
@test F.A ≈ I(n) + (A_'*(I + A_*dt/2))*dt
@test F.B ≈ (A_'B_*dt/2 + B_)*dt


## Test an affine system
struct DLTI2{TA,TB,TD} <: DiscreteLTI
    A::TA
    B::TB
    d::TD
end
RobotDynamics.state_dim(::DLTI2) = 15 
RobotDynamics.control_dim(::DLTI2) = 10
RobotDynamics.get_A(model::DLTI2) = model.A
RobotDynamics.get_B(model::DLTI2) = model.B
RobotDynamics.get_d(model::DLTI2) = model.d
RobotDynamics.is_affine(::DLTI2) = Val(true)

d = @SVector rand(n)
model = DLTI2(SMatrix{n,n}(Ad), SMatrix{n,m}(Bd), d)
x2 = discrete_dynamics(DiscreteSystemQuadrature, model, z)
@test x2 ≈ Ad*x + Bd*u + d
@test (@ballocated discrete_dynamics(DiscreteSystemQuadrature, $model, $z)) == 0

F = RobotDynamics.DynamicsJacobian(model)
discrete_jacobian!(DiscreteSystemQuadrature, F, model, z)
@test F ≈ [Ad Bd]