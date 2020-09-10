using RobotDynamics
using StaticArrays
using BenchmarkTools

const RD = RobotDynamics

nonlinear_model = Cartpole()

N = 15
dt = 0.01
n = state_dim(nonlinear_model)
m = control_dim(nonlinear_model)

times = 0:dt:(N-1)*dt
trajectory = Traj(n, m, dt, N)

for i=1:N
    knot_point = trajectory[i]
    local x, u = rand(nonlinear_model)
    RD.set_state!(knot_point, x)
    RD.set_control!(knot_point, u)
end

@create_discrete_ltv(LinearizedCartpoleTV, n, m, N, true)
linear_model = LinearizedCartpoleTV()
RD.set_times!(linear_model, times)

# @test (@ballocated RD.linearize_and_discretize!(Exponential, $linear_model, $nonlinear_model, $trajectory)) == 0
# @test (@ballocated RD.linearize_and_discretize!($linear_model, $nonlinear_model, $trajectory)) == 0

RD.linearize_and_discretize!(linear_model, nonlinear_model, trajectory)

for i=1:N-1
    knot_point = trajectory[i]
    @test discrete_dynamics(RK3, nonlinear_model, knot_point) ≈ discrete_dynamics(DiscreteSystemQuadrature, linear_model, knot_point) atol=1e-12

    σ = 1e-3
    knot_point.z += σ*randn(n+m)

    @test discrete_dynamics(RK3, nonlinear_model, knot_point) ≈ discrete_dynamics(DiscreteSystemQuadrature, linear_model, knot_point) atol=1e-2
end

###################################
# Exponential Discretization Tests
using ControlSystems

dt = 0.01
C_111 = ss([-5], [2], [3], [0])
D_111 = c2d(C_111, dt, :zoh)
@create_continuous_lti(C_111_model, 1, 1)
C_111_rd = C_111_model()
set_A!(C_111_rd, [-5])
set_B!(C_111_rd, [2])

@create_discrete_lti(D_111_model, 1, 1)
D_111_rd = D_111_model()
discretize!(Exponential, D_111_rd, C_111_rd, dt=dt)
@test D_111[1].A ≈ get_A(D_111_rd)
@test D_111[1].B ≈ get_B(D_111_rd)

####

C_212 = ss([-5 -3; 2 -9], [1; 2], [1 0; 0 1], [0; 0])
D_212 = c2d(C_212, dt, :zoh)
@create_continuous_lti(C_212_model, 2, 1)
C_212_rd = C_212_model()
set_A!(C_212_rd, [-5 -3; 2 -9])
set_B!(C_212_rd, [1; 2])

@create_discrete_lti(D_212_model, 2, 1)
D_212_rd = D_212_model()
discretize!(Exponential, D_212_rd, C_212_rd, dt=dt)
@test D_212[1].A ≈ get_A(D_212_rd)
@test D_212[1].B ≈ get_B(D_212_rd)

###
C_221 = ss([-5 -3; 2 -9], [1 0; 0 2], [1 0], [0 0])
D_221 = c2d(C_221, dt, :zoh)
@create_continuous_lti(C_221_model, 2, 2)
C_221_rd = C_221_model()
set_A!(C_221_rd, [-5 -3; 2 -9])
set_B!(C_221_rd, [1 0; 0 2])

@create_discrete_lti(D_221_model, 2, 2)
D_221_rd = D_221_model()
discretize!(Exponential, D_221_rd, C_221_rd, dt=dt)
@test D_221[1].A ≈ get_A(D_221_rd)
@test D_221[1].B ≈ get_B(D_221_rd)