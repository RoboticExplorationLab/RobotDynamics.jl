using RobotDynamics
using ForwardDiff
using RobotZoo
using LinearAlgebra

model = RobotZoo.Cartpole()
x,u = rand(model)
n,m = size(model)
dt = 0.01
t = 0.0
z = KnotPoint(x,u,dt,t)
k1(x) = dynamics(model, x,             u, t       )*dt;
k2(x) = dynamics(model, x + k1(x)/2,      u, t + dt/2)*dt;
k3(x) = dynamics(model, x - k1(x) + 2*k2(x), u, t + dt  )*dt;
xnext(x) = x + (k1(x) + 4*k2(x) + k3(x))/6
k1_(u) = dynamics(model, x,             u, t       )*dt;
k2_(u) = dynamics(model, x + k1_(u)/2,      u, t + dt/2)*dt;
k3_(u) = dynamics(model, x - k1_(u) + 2*k2_(u), u, t + dt  )*dt;
xnext_(u) = x + (k1_(u) + 4*k2_(u) + k3_(u))/6
k1(x) == k1_(u)
k2(x) == k2_(u)
k3(x) == k3_(u)
xnext(x) == xnext_(u)

F = zeros(n,n+m)
jacobian!(F, model, z)
A1 = F[z._x, z._x]
B1 = F[z._x, z._u]
RobotDynamics.set_state!(z, x + k1(x)/2)
jacobian!(F, model, z)
A2 = F[z._x, z._x]
B2 = F[z._x, z._u]
RobotDynamics.set_state!(z, x - k1(x) + 2*k2(x))
jacobian!(F, model, z)
A3 = F[z._x, z._x]
B3 = F[z._x, z._u]


ForwardDiff.jacobian(k1,x) ≈ A1*dt
ForwardDiff.jacobian(k2,x) ≈ A2*(I + 0.5*A1*dt)*dt
ForwardDiff.jacobian(k3,x) ≈ A3*(I - A1*dt + 2*A2*(I + 0.5*A1*dt)*dt)*dt
ForwardDiff.jacobian(xnext,x) ≈ I +
        (A1*dt +
        4*A2*(I + 0.5*A1*dt)*dt +
        A3*(I - A1*dt + 2*A2*(I + 0.5*A1*dt)*dt)*dt)/6

ForwardDiff.jacobian(k1_,u) ≈ B1*dt
ForwardDiff.jacobian(k2_,u) ≈ (0.5*A2*B1*dt + B2)*dt
ForwardDiff.jacobian(k3_,u) ≈ (A3*(-B1*dt + A2*B1*dt*dt + 2*B2*dt) + B3)*dt
ForwardDiff.jacobian(xnext_,u) ≈
        (B1*dt +
        4*(0.5*A2*B1*dt + B2)*dt +
        (A3*(-B1*dt + A2*B1*dt*dt + 2*B2*dt) + B3)*dt)/6

fu(u) = discrete_dynamics(RK3, model, x, u, t, dt)
fu(u) ≈ discrete_dynamics(RK3, model, z)
ForwardDiff.jacobian(fu, u)
(B1 + 4*B2 + B3)*dt/6

z = KnotPoint(x,u,dt,t)
discrete_jacobian!(RK3, F, model, z)
F[1:n,1:n] ≈ ForwardDiff.jacobian(xnext,x)
F[1:n,n .+ (1:m)]
(B1 + 4*B2 + B3)*dt/6
