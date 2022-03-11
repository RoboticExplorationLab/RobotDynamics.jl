using RobotDynamics
using ForwardDiff
using LinearAlgebra

using RobotDynamics: dynamics, discrete_dynamics

function check_jacobians(model, z)
    t,dt, = z.t, z.dt
    xn = zeros(RD.state_dim(model))

    J0 = ForwardDiff.jacobian(z->RD.discrete_dynamics(model, z[1:4], z[5:5], t, dt), z.z)
    J = similar(J0)
    for sig in (RD.StaticReturn(), RD.InPlace()), diff in (RD.ForwardAD(), RD.FiniteDifference(), RD.UserDefined())
        atol = diff == RD.FiniteDifference() ? 1e-6 : 1e-10
        RD.jacobian!(sig, diff, model, J, xn, z)
        @test J ≈ J0 atol=atol
    end
end

cmodel = Cartpole()
x,u = rand(cmodel)
n,m = RD.dims(cmodel)
dt = 0.01
t = 0.0
z = RD.KnotPoint(x,u,t,dt)

# Check the Cartpole analytical Jacobian 
J = zeros(n, n+m)
J0 = zero(J)
xdot = zeros(n)
RD.jacobian!(RD.StaticReturn(), RD.ForwardAD(), cmodel, J, xdot, z)
RD.jacobian!(RD.StaticReturn(), RD.UserDefined(), cmodel, J0, xdot, z)
@test J0 ≈ J

# Test Euler
model = RD.DiscretizedDynamics{RD.Euler}(cmodel)
xdot = RD.dynamics(cmodel, z)
@test discrete_dynamics(model, z) ≈ x + xdot * dt

check_jacobians(model, z)

# # Test RK2
# _k1 = dynamics(cmodel, x, u)*dt
# _k2 = dynamics(cmodel, x + _k1/2, u)*dt
# @test discrete_dynamics(model, z) ≈ x + _k2

# Test RK4
model = RD.DiscretizedDynamics{RD.RK4}(cmodel)
_k1 = dynamics(cmodel, x, u)*dt
_k2 = dynamics(cmodel, x + _k1/2, u)*dt
_k3 = dynamics(cmodel, x + _k2/2, u)*dt
_k4 = dynamics(cmodel, x + _k3, u)*dt

@test discrete_dynamics(model, z) ≈ x + (_k1 + 2*_k2 + 2*_k3 + _k4)/6 
xn = zeros(n)
RD.discrete_dynamics!(model, xn, z)
@test xn ≈ x + (_k1 + 2*_k2 + 2*_k3 + _k4)/6 
RD.discrete_dynamics!(model, xn, x, u, t, dt)
@test xn ≈ x + (_k1 + 2*_k2 + 2*_k3 + _k4)/6 

check_jacobians(model, z)
J0 = ForwardDiff.jacobian(z->RD.discrete_dynamics(model, z[1:4], z[5:5], t, dt), z.z)
J = similar(J0)
RD.jacobian!(RD.InPlace(), RD.ForwardAD(), model, J, xn, z)
@test J ≈ J0
RD.jacobian!(RD.StaticReturn(), RD.UserDefined(), model, J, xn, z)
@test J ≈ J0

RD.jacobian!(RD.StaticReturn(), RD.UserDefined(), model, J, xn, z)
RD.jacobian!(RD.InPlace(), RD.UserDefined(), model, J, xn, z)
@test J ≈ J0

int1 = RD.RK4(cmodel)
int2 = copy(int1)
@test int1.k1 !== int2.k1
@test int1.k2 !== int2.k2
@test int1.A !== int2.A

# Test RK3 and jacobian
model = RD.DiscretizedDynamics{RD.RK3}(cmodel)
k1(x) = dynamics(cmodel, x,                   u, t       )*dt;
k2(x) = dynamics(cmodel, x + k1(x)/2,         u, t + dt/2)*dt;
k3(x) = dynamics(cmodel, x - k1(x) + 2*k2(x), u, t + dt  )*dt;
xnext(x) = x + (k1(x) + 4*k2(x) + k3(x))/6
k1_(u) = dynamics(cmodel, x,             u, t       )*dt;
k2_(u) = dynamics(cmodel, x + k1_(u)/2,      u, t + dt/2)*dt;
k3_(u) = dynamics(cmodel, x - k1_(u) + 2*k2_(u), u, t + dt  )*dt;
xnext_(u) = x + (k1_(u) + 4*k2_(u) + k3_(u))/6

# Make sure the test functions match
@test k1(x) == k1_(u)
@test k2(x) == k2_(u)
@test k3(x) == k3_(u)
@test xnext(x) == xnext_(u)

# Check actual implementation
@test xnext(x) ≈ RD.discrete_dynamics(model, z)
@test xnext_(u) ≈ RD.discrete_dynamics(model, z)
@test xnext(x) ≈ RD.discrete_dynamics(model, x, u, t, dt)

# Check Jacobian
A = ForwardDiff.jacobian(xnext, x)
B = ForwardDiff.jacobian(xnext_, u)
J0 = [A B]
J = zeros(n,n+m)
RD.jacobian!(RD.StaticReturn(), RD.UserDefined(), model, J, xn, z)
@test J ≈ J0
RD.jacobian!(RD.InPlace(), RD.UserDefined(), model, J, xn, z)
@test J ≈ J0

check_jacobians(model, z)