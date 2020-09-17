using RobotDynamics
using LinearAlgebra
using StaticArrays
using Test
using BenchmarkTools
using Rotations

function LinearQuad(model::Quadrotor, x0, u0, dt::Real)
    F = RobotDynamics.DynamicsJacobian(model)
    z = KnotPoint(x0,u0,dt)
    discrete_jacobian!(RK3, F, model, z) 
    G = @MMatrix zeros(n,n-1)
    RobotDynamics.state_diff_jacobian!(G, model, z)
    A = RobotDynamics.get_A(F)
    B = RobotDynamics.get_B(F)
    A = G'A*G
    B = G'B
    LinearModel(SMatrix(A), SMatrix(B), dt=dt)
end

# Create model
quad = Quadrotor()
n = state_dim(quad)
x0,u0 = zeros(quad)
dt = 0.01
t = 0.0

model = LinearQuad(quad, x0, u0, dt)
A = model.A[1]
B = model.B[1]
x,u = rand(model)
z = KnotPoint(x,u,dt,t)

# Test basic properties
@test RobotDynamics.is_timevarying(model) == false
@test RobotDynamics.is_affine(model) == false
@test size(B) == (state_dim(model), control_dim(model))
@test size(A) == (state_dim(model), state_dim(model))

# Continuous dynamics shouldn't be implemented
@test_throws AssertionError dynamics(model, x, u)
@test_throws AssertionError dynamics(model, x, u, 0.)
@test_throws AssertionError dynamics(model, z) 
F = RobotDynamics.DynamicsJacobian(model)
@test_throws MethodError jacobian!(F, model, x, u)
@test_throws MethodError jacobian!(F, model, x, u, 0)
@test_throws AssertionError jacobian!(F, model, z) 

# Test discrete dynamics
x2 = discrete_dynamics(PassThrough, model, x, u, t, dt)
@test x2 ≈ A*x + B*u
@test (@ballocated discrete_dynamics(PassThrough, $model, $x, $u, $t, $dt)) == 0

discrete_jacobian!(PassThrough, F, model, z)
@test F ≈ [A B]
@test (@ballocated discrete_jacobian!(PassThrough, $F, $model, $z) samples=2 evals=2) == 0

@test_throws AssertionError discrete_dynamics(RK3, model, x, u, t, dt)
@test_throws AssertionError discrete_jacobian!(RK3, F, model, z)