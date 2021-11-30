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

## Test RK4
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

## Implicit Midpoint
model = RD.DiscretizedDynamics{RD.ImplicitMidpoint}(cmodel)
z1 = KnotPoint(randn(model)..., t, h) 
z2 = KnotPoint(randn(model)..., t+h, h) 
x1,u1 = RD.state(z1), RD.control(z1)
x2,u2 = RD.state(z2), RD.control(z2)
y1,y2 = zeros(n), zeros(n)
J1,J2 = zeros(n,n+m), zeros(n,n+m)
J10,J20 = zeros(n,n+m), zeros(n,n+m)

for i = 1:2
    if i == 2
        z1 = KnotPoint{n,m}(Vector(z1.z), t,h)
        z2 = KnotPoint{n,m}(Vector(z2.z), t+h,h)
    end

    e_mid = x1 - x2 + h*RD.dynamics(cmodel, (x1 + x2)/2, u1, 0.0) 
    @test e_mid ≈ RD.dynamics_error(model, z2, z1)
    RD.dynamics_error!(model, y2, y1, z2, z1)
    @test e_mid ≈ y2

    RD.dynamics_error_jacobian!(StaticReturn(), RD.UserDefined(), model, J20, J10, y2, y1, z2, z1)
    RD.dynamics_error_jacobian!(StaticReturn(), ForwardAD(), model, J2, J1, y2, y1, z2, z1)
    @test J2 ≈ J20
    @test J1 ≈ J10

    RD.dynamics_error_jacobian!(InPlace(), RD.UserDefined(), model, J20, J10, y2, y1, z2, z1)
    RD.dynamics_error_jacobian!(InPlace(), ForwardAD(), model, J2, J1, y2, y1, z2, z1)
    @test J2 ≈ J20
    @test J1 ≈ J10

    RD.dynamics_error_jacobian!(StaticReturn(), FiniteDifference(), model, J2, J1, y2, y1, z2, z1)
    @test J2 ≈ J20
    @test J1 ≈ J10
    RD.dynamics_error_jacobian!(InPlace(), FiniteDifference(), model, J2, J1, y2, y1, z2, z1)
    @test J2 ≈ J20
    @test J1 ≈ J10
end

test_error_allocs(model)