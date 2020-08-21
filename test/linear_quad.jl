using RobotDynamics
using LinearAlgebra
using StaticArrays
using Test
using BenchmarkTools
using Rotations


struct LinearQuad{T} <: DiscreteLTI
    A::SMatrix{12,12,T,144}    
    B::SMatrix{12,4,T,48}
end 
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
    LinearQuad(SMatrix(A), SMatrix(B))
end
RobotDynamics.get_A(model::LinearQuad) = model.A
RobotDynamics.get_B(model::LinearQuad) = model.B
RobotDynamics.state_dim(::LinearQuad) = 12
RobotDynamics.control_dim(::LinearQuad) = 4


# Create model
quad = Quadrotor()
n = state_dim(quad)
x0,u0 = zeros(quad)
dt = 0.01
t = 0.0

model = LinearQuad(quad, x0, u0, dt)
A = RobotDynamics.get_A(model)
B = RobotDynamics.get_B(model)
x,u = rand(model)
z = KnotPoint(x,u,dt,t)

# Test basic properties
@test RobotDynamics.is_time_varying(model) == false
@test RobotDynamics.is_affine(model) == Val(false)
@test size(RobotDynamics.get_B(model)) == (state_dim(model), control_dim(model))
@test RobotDynamics.get_A(model, 10) === A
@test RobotDynamics.get_B(model, 10) === B

# Continuous dynamics shouldn't be implemented
@test_throws MethodError dynamics(model, x, u)
@test_throws MethodError dynamics(model, x, u, 0.)
@test_throws MethodError dynamics(model, z) 
F = RobotDynamics.DynamicsJacobian(model)
@test_throws MethodError jacobian!(F, model, x, u)
@test_throws MethodError jacobian!(F, model, x, u, 0)
@test_throws MethodError jacobian!(F, model, z) 

# Test discrete dynamics
x2 = discrete_dynamics(DiscreteSystemQuadrature, model, x, u, t, dt)
@test x2 ≈ A*x + B*u
@test (@ballocated discrete_dynamics(DiscreteSystemQuadrature, $model, $x, $u, $t, $dt)) == 0
x2 = discrete_dynamics(DiscreteSystemQuadrature, model, z) 
@test x2 ≈ A*x + B*u
@test x2 isa SVector{12}

discrete_jacobian!(DiscreteSystemQuadrature, F, model, z)
@test F ≈ [A B]
@test (@ballocated discrete_jacobian!(DiscreteSystemQuadrature, $F, $model, $z) samples=2 evals=2) == 0

@test_throws MethodError discrete_dynamics(RK3, model, x, u, t, dt)
@test_throws MethodError discrete_jacobian!(RK3, F, model, z)