using RobotDynamics
using StaticArrays
using BenchmarkTools

const RD = RobotDynamics

nonlinear_model = Cartpole()

N = 5
dt = 0.01
n = state_dim(nonlinear_model)
m = control_dim(nonlinear_model)

trajectory = RD.Traj(n, m, dt, N)
times = RD.get_times(trajectory)

for knot_point in trajectory
    knot_point.z = @SVector rand(n+m)
end

linearized_model = LinearizedModel(nonlinear_model, trajectory, RK3; is_affine=true)

for i=1:N-1
    knot_point = trajectory[i]
    @test discrete_dynamics(RK3, nonlinear_model, knot_point) ≈ discrete_dynamics(PassThrough, linearized_model, knot_point) atol=1e-4

    σ = 1e-6
    knot_point.z += σ*randn(n+m)

    @test discrete_dynamics(RK3, nonlinear_model, knot_point) ≈ discrete_dynamics(PassThrough, linearized_model, knot_point) atol=σ*1e2
end

for knot_point in trajectory
    knot_point.z = @SVector rand(n+m)
end

update_trajectory!(linearized_model, trajectory, RK3)

for i=1:N-1
    knot_point = trajectory[i]
    @test discrete_dynamics(RK3, nonlinear_model, knot_point) ≈ discrete_dynamics(PassThrough, linearized_model, knot_point) atol=1e-4

    σ = 1e-6
    knot_point.z += σ*randn(n+m)

    @test discrete_dynamics(RK3, nonlinear_model, knot_point) ≈ discrete_dynamics(PassThrough, linearized_model, knot_point) atol=σ*1e2
end

##################################
# Exponential Discretization Tests
using ControlSystems
using StaticArrays

dt = 0.01
C_111 = ss([-5], [2], [3], [0])
D_111 = c2d(C_111, dt, :zoh)
C_111_rd = LinearModel(-5*ones(1,1), 2*ones(1,1))
D_111_rd = LinearModel(1, 1, dt=dt)

RD.discretize!(RD.Exponential, D_111_rd, C_111_rd)
@test D_111[1].A ≈ D_111_rd.A[1]
@test D_111[1].B ≈ D_111_rd.B[1]

####

C_212 = ss([-5 -3; 2 -9], [1; 2], [1 0; 0 1], [0; 0])
D_212 = c2d(C_212, dt, :zoh)
C_212_rd = LinearModel([-5 -3; 2 -9], reshape([1; 2], 2, 1))
D_212_rd = LinearModel(2, 1, dt=dt)

RD.discretize!(RD.Exponential, D_212_rd, C_212_rd)
@test D_212[1].A ≈ D_212_rd.A[1]
@test D_212[1].B ≈ D_212_rd.B[1]

###
C_221 = ss([-5 -3; 2 -9], [1 0; 0 2], [1 0], [0 0])
D_221 = c2d(C_221, dt, :zoh)
C_221_rd = LinearModel([-5 -3; 2 -9], [1 0; 0 2])
D_221_rd = LinearModel(2, 2, dt=dt)

RD.discretize!(RD.Exponential, D_221_rd, C_221_rd)
@test D_221[1].A ≈ D_221_rd.A[1]
@test D_221[1].B ≈ D_221_rd.B[1]

# Constant Discrete Jacobian Tests:
include("random_linear.jl")

n,m = 5, 5
A, B = gencontinuous(n,m)
random_model = LinearModel(A, B)
random_model_discrete = LinearModel(n, m; dt=dt)
x,u = rand(random_model)
z = KnotPoint(x, u, dt)

struct RandomLinear{n,m,T} <: AbstractModel
    A::SMatrix{n,n,T}
    B::SMatrix{n,m,T}
end
RD.state_dim(::RandomLinear{n}) = n
RD.control_dim(::RandomLinear{<:Any, m}) = m
RD.dynamics(model::RandomLinear, x, u) = model.A*x + model.B*u

random_model_test = RandomLinear{n,m,Float64}(A,B)

F = RD.DynamicsJacobian(n,m)

discretize!(RK2, random_model_discrete, random_model)
discrete_jacobian!(RK2, F, random_model_test, z)

@test RD.get_A(F) ≈ random_model_discrete.A[1]
@test RD.get_B(F) ≈ random_model_discrete.B[1]

discretize!(RK3, random_model_discrete, random_model)
discrete_jacobian!(RK3, F, random_model_test, z)

@test RD.get_A(F) ≈ random_model_discrete.A[1]
@test RD.get_B(F) ≈ random_model_discrete.B[1]

discretize!(RK4, random_model_discrete, random_model)
discrete_jacobian!(RK4, F, random_model_test, z)

@test RD.get_A(F) ≈ random_model_discrete.A[1]
@test RD.get_B(F) ≈ random_model_discrete.B[1]