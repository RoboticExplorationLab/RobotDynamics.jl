using Test
using LinearAlgebra, StaticArrays
using BenchmarkTools
using RobotDynamics
using RobotDynamics: LinearModel
const RD = RobotDynamics
include("random_linear.jl")

##
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

@btime discrete_dynamics($model, $z)
@btime dynamics($model, $z)


model = LinearModel(A,B,use_static=false)
@test model.use_static == false
@test RD.is_discrete(model) == false
@test RD.is_timevarying(model) == false
@test RD.is_affine(model) == false
@test size(model) == (n,m) 

@test dynamics(model, z) ≈ A*x + B*u
@test discrete_dynamics(RK2, model, z) ≈ x2