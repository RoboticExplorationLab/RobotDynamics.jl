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

linear_model = @create_discrete_ltv(LinearizedCartpoleTV, n, m, N, true)
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