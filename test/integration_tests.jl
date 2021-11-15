using RobotDynamics
using ForwardDiff
using LinearAlgebra

using RobotDynamics: dynamics, discrete_dynamics

cmodel = Cartpole()
x,u = rand(model)
n,m = size(model)
dt = 0.01
t = 0.0
z = KnotPoint(x,u,t,dt)

# Test Euler
model = RD.DiscretizedDynamics{RD.Euler}(cmodel)
xdot = RD.dynamics(cmodel, z)
@test discrete_dynamics(model, z) ≈ x + xdot * dt

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

J0 = ForwardDiff.jacobian(z->RD.discrete_dynamics(model, z[1:4], z[5:5], t, dt), z.z)
J = similar(J0)
RD.jacobian!(RD.InPlace(), RD.ForwardAD(), model, J, xn, z)
@test J ≈ J0

# # Test RK3 and jacobian
# k1(x) = dynamics(model, x,                   u, t       )*dt;
# k2(x) = dynamics(model, x + k1(x)/2,         u, t + dt/2)*dt;
# k3(x) = dynamics(model, x - k1(x) + 2*k2(x), u, t + dt  )*dt;
# xnext(x) = x + (k1(x) + 4*k2(x) + k3(x))/6
# k1_(u) = dynamics(model, x,             u, t       )*dt;
# k2_(u) = dynamics(model, x + k1_(u)/2,      u, t + dt/2)*dt;
# k3_(u) = dynamics(model, x - k1_(u) + 2*k2_(u), u, t + dt  )*dt;
# xnext_(u) = x + (k1_(u) + 4*k2_(u) + k3_(u))/6

# # Make sure the test functions match
# @test k1(x) == k1_(u)
# @test k2(x) == k2_(u)
# @test k3(x) == k3_(u)
# @test xnext(x) == xnext_(u)

# # Evaluate at each of the intermediate states
# F = RD.DynamicsJacobian(model)
# RD.set_state!(z, x)
# jacobian!(F, model, z)
# A1 = RD.get_static_A(F) 
# B1 = RD.get_static_B(F) 
# RobotDynamics.set_state!(z, x + k1(x)/2)
# jacobian!(F, model, z)
# A2 = RD.get_static_A(F) 
# B2 = RD.get_static_B(F) 
# RobotDynamics.set_state!(z, x - k1(x) + 2*k2(x))
# jacobian!(F, model, z)
# A3 = RD.get_static_A(F) 
# B3 = RD.get_static_B(F) 

# # Test analytical formulas
# @test ForwardDiff.jacobian(k1,x) ≈ A1*dt
# @test ForwardDiff.jacobian(k2,x) ≈ A2*(I + 0.5*A1*dt)*dt
# @test ForwardDiff.jacobian(k3,x) ≈ A3*(I - A1*dt + 2*A2*(I + 0.5*A1*dt)*dt)*dt
# @test ForwardDiff.jacobian(xnext,x) ≈ I +
#         (A1*dt +
#         4*A2*(I + 0.5*A1*dt)*dt +
#         A3*(I - A1*dt + 2*A2*(I + 0.5*A1*dt)*dt)*dt)/6

# @test ForwardDiff.jacobian(k1_,u) ≈ B1*dt
# @test ForwardDiff.jacobian(k2_,u) ≈ (0.5*A2*B1*dt + B2)*dt
# @test ForwardDiff.jacobian(k3_,u) ≈ (A3*(-B1*dt + A2*B1*dt*dt + 2*B2*dt) + B3)*dt
# @test ForwardDiff.jacobian(xnext_,u) ≈
#         (B1*dt +
#         4*(0.5*A2*B1*dt + B2)*dt +
#         (A3*(-B1*dt + A2*B1*dt*dt + 2*B2*dt) + B3)*dt)/6

# # Test actual implemented function
# z = KnotPoint(x,u,dt,t)
# fu(u) = discrete_dynamics(RK3, model, x, u, t, dt)
# @test fu(u) ≈ discrete_dynamics(RK3, model, z)
# ForwardDiff.jacobian(fu, u)
# (B1 + 4*B2 + B3)*dt/6

# # Forward Diff through the dynamics
# D = RobotDynamics.DynamicsJacobian(n,m)
# discrete_jacobian!(RK3, D, model, z)
# @test ForwardDiff.jacobian(xnext, x) ≈ D.A
# @test ForwardDiff.jacobian(xnext_, u) ≈ D.B

# # Use continuous Jacobian
# tmp = [RobotDynamics.DynamicsJacobian(n,m) for k = 1:3]
# jacobian!(RK3, D, model, z, tmp)
# @test ForwardDiff.jacobian(xnext, x) ≈ D.A
# @test ForwardDiff.jacobian(xnext_, u) ≈ D.B

# @btime discrete_jacobian!(RK3, $D, $model, $z)
# @btime jacobian!(RK3, $D, $model, $z, $tmp)
