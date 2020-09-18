using Test
using LinearAlgebra, StaticArrays
using BenchmarkTools
using RobotDynamics
using RobotDynamics: LinearModel
const RD = RobotDynamics
include("random_linear.jl")

## Continuous LTI
n,m = 10,5
dt = 0.1
t = 0.0
A,B = gencontrollable(n,m)
model = LinearModel(A,B)
@test model.use_static == true
@test RD.is_discrete(model) == false
@test RD.is_timevarying(model) == false
@test RD.is_affine(model) == false
@test size(model) == (n,m) 

x,u = rand(model)
z = KnotPoint(x,u,dt,t)
length(x) == n
length(u) == m
@test dynamics(model, x, u) ≈ A*x + B*u
@test_throws AssertionError discrete_dynamics(RD.PassThrough, model, x, u, t, dt)
x2 = x + (A*(x + (A*x + B*u)*dt/2) + B*u)*dt
@test discrete_dynamics(RK2, model, z) ≈ x2

@test @ballocated(discrete_dynamics($model, $z)) == 0
@test @ballocated(dynamics($model, $z)) == 0

F = RD.DynamicsJacobian(n,m)
kp = KnotPoint(x,u,dt,t)
@test_throws AssertionError discrete_jacobian!(PassThrough, F, model, kp)
jacobian!(F, model, kp)
@test RD.get_A(F) == A
@test RD.get_B(F) == B


model = LinearModel(A,B,use_static=false)
@test model.use_static == false
@test RD.is_discrete(model) == false
@test RD.is_timevarying(model) == false
@test RD.is_affine(model) == false
@test size(model) == (n,m) 

@test dynamics(model, z) ≈ A*x + B*u
@test discrete_dynamics(RK2, model, z) ≈ x2

@test @ballocated(dynamics($model, $z)) == 0
@test @ballocated(discrete_dynamics(RK2, $model, $z)) == 0 

# affine
d = randn(n)
model = LinearModel(A,B,d)
@test RD.is_affine(model) == true
@test RD.is_timevarying(model) == false
@test RD.is_discrete(model) == false
@test dynamics(model, z) ≈ A*x + B*u + d

# Discrete LTI
model = LinearModel(A,B,d;dt=dt)
@test RD.is_affine(model) == true
@test RD.is_timevarying(model) == false
@test RD.is_discrete(model) == true

@test_throws AssertionError dynamics(model, z) 
@test discrete_dynamics(RD.PassThrough, model, x, u, t, dt) ≈ A*x + B*u + d
