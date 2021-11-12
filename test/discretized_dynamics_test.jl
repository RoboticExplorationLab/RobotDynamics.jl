include("double_integrator.jl")

## Test Euler
@autodiff struct DoubleIntegrator{D} <: DI{D} end
cmodel = DoubleIntegrator{2}()
n,m = size(cmodel)

# Evaluate Euler step explicitly
x0,u0 = randn(cmodel)
t,h = 0.0, 0.1
xn = zeros(n)
z0 = KnotPoint(x0,u0,t,h)
xdot = RD.dynamics(cmodel, z0)
x2 = x0 + h * xdot

# Create discretized model
model = RD.DiscretizedDynamics{RD.Euler}(cmodel)
@test x2 ≈ RD.discrete_dynamics(model, z0)
RD.discrete_dynamics!(model, xn, z0)
@test x2 ≈ xn

test_model(model)

# Test RK4
x,u = x0, u0
k1 = RD.dynamics(cmodel, x,        u, t      )*h
k2 = RD.dynamics(cmodel, x + k1/2, u, t + h/2)*h
k3 = RD.dynamics(cmodel, x + k2/2, u, t + h/2)*h
k4 = RD.dynamics(cmodel, x + k3,   u, t + h  )*h
x_rk4 = x + (k1 + 2k2 + 2k3 + k4)/6

model = RD.DiscretizedDynamics{RD.RK4}(cmodel)
@test x_rk4 ≈ RD.discrete_dynamics(model, z0)
RD.discrete_dynamics!(model, xn, z0)
@test x_rk4 ≈ xn

test_model(model)
test_error_allocs(model)