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
@test RD.dims(model) == (n,m,n) 

x,u = rand(model)
z = KnotPoint(x,u,dt,t)
@test length(x) == n
@test length(u) == m
@test dynamics(model, x, u) ≈ A*x + B*u
@test_throws AssertionError discrete_dynamics(RD.PassThrough, model, x, u, t, dt)
x2 = x + (A*(x + (A*x + B*u)*dt/2) + B*u)*dt
@test discrete_dynamics(RK2, model, z) ≈ x2

@test @ballocated(discrete_dynamics($model, $z), samples=2, evals=2) == 0
@test @ballocated(dynamics($model, $z), samples=2, evals=2) == 0

F = RD.DynamicsJacobian(n,m)
kp = KnotPoint(x,u,dt,t)
@test_throws AssertionError discrete_jacobian!(PassThrough, F, model, kp)
jacobian!(F, model, kp)
@test RD.get_A(F) == A
@test RD.get_B(F) == B

## Non-static model
model = LinearModel(A,B,use_static=false)
@test model.use_static == false
@test RD.is_discrete(model) == false
@test RD.is_timevarying(model) == false
@test RD.is_affine(model) == false
@test RD.dims(model) == (n,m,n) 

@test dynamics(model, z) ≈ A*x + B*u
@test discrete_dynamics(RK2, model, z) ≈ x2

@test @ballocated(dynamics($model, $z), samples=2, evals=2) == 0
@test @ballocated(discrete_dynamics(RK2, $model, $z), samples=2, evals=2) == 0 

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


## Continuous LTV
N = 11
tf = dt*(N-1)
t = 0.5
AB = [gencontrollable(n,m) for k = 1:N]
A = [A for (A,B) in AB]
B = [B for (A,B) in AB]
times = range(0,tf, length=N)
k = findfirst(times .== t)

@test_throws AssertionError LinearModel(A,B)
model = LinearModel(A,B, times=times)
x,u = rand(model)
@test length(x) == n
@test length(u) == m
z = KnotPoint(x,u,dt,t)

@test dynamics(model, z) ≈ A[k]*x + B[k]*u
F = RD.DynamicsJacobian(model)
jacobian!(F, model, z)
@test F ≈ [A[k] B[k]]

@test RD.is_discrete(model) == false
@test RD.is_affine(model) == false
@test RD.is_timevarying(model) == true

x2 = x + (A[k]*(x + (A[k]*x + B[k]*u)*dt/2) + B[k]*u)*dt
@test discrete_dynamics(RK2, model, z) ≈ x2

## Affine Discrete LTV
d = [@SVector rand(n) for k = 1:N]
model = LinearModel(A,B,d, times=times, dt=dt)
@test_throws AssertionError dynamics(model, z)
@test discrete_dynamics(PassThrough, model, z) ≈ A[k]*x + B[k]*u + d[k]

discrete_jacobian!(PassThrough, F, model, z)
@test F ≈ [A[k] B[k]]

@test_throws AssertionError discrete_dynamics(RK2, model, z)

@test RD.is_discrete(model) == true
@test RD.is_affine(model) == true
@test RD.is_timevarying(model) == true


## Varying time steps
dts = normalize(rand(N-1), 1)*tf
times = insert!(cumsum(dts), 1, 0)
linmodel = LinearModel(A, B, times=times, dt=NaN)
@test RD.is_discrete(linmodel)

@test discrete_dynamics(PassThrough, linmodel, x, u, 0.0, dts[1]) ≈ A[1]*x + B[1]*u
@test discrete_dynamics(PassThrough, linmodel, x, u, times[2], dts[2]) ≈ A[2]*x + B[2]*u
@test_throws AssertionError discrete_dynamics(PassThrough, linmodel, x, u, 0.0, dts[2])
