using RobotDynamics
using StaticArrays

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
    x, u = rand(nonlinear_model)
    RD.set_state!(knot_point, x)
    RD.set_control!(knot_point, u)
end

linear_model = @create_discrete_ltv(LinearizedCartpole, n, m, N, true)
RD.set_times!(linear_model, times)

RD.linearize_and_discretize!(linear_model, nonlinear_model, trajectory)

for i=1:N-1
    knot_point = trajectory[i]
    @test discrete_dynamics(RK3, nonlinear_model, knot_point) ≈ discrete_dynamics(DiscreteSystemQuadrature, linear_model, knot_point)
end