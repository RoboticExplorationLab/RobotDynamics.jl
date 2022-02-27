using RobotDynamics
using Test
using StaticArrays

struct MyModel <: RD.ContinuousDynamics end
model = MyModel()
@test_throws RD.NotImplementedError RD.state_dim(model)
@test_throws RD.NotImplementedError RD.control_dim(model)

model = Cartpole()
x,u = zeros(model)
@test sum(x) == 0
@test sum(u) == 0
x,u = rand(model)
@test sum(x) != 0
@test sum(u) != 0
xdot = RD.dynamics(model, x, u)
@test sum(x) != 0
xdot = RD.dynamics(model, x, u, 0.0)
@test sum(x) != 0
y = zeros(length(x))
RD.dynamics!(model, y, x, u)
@test y ≈ xdot
RD.dynamics!(model, y, x, u, 0.0)
@test y ≈ xdot

x,u = zeros(Float32,model)
@test x isa SVector{4,Float32}
@test u isa SVector{1,Float32}

x,u = rand(Float32,model)
@test x isa SVector{4,Float32}
@test u isa SVector{1,Float32}

x,u = fill(model, 10.0)
@test x ≈ @SVector fill(10,4)
@test u ≈ @SVector fill(10,1)


n,m = RD.dims(model)
t,dt = 0, 0.1
F = zeros(n,n+m)
y = zeros(n)
z = RD.KnotPoint(x,u,t,dt)
RD.jacobian!(RD.StaticReturn(), RD.ForwardAD(), model, F, y, z)
@test sum(F) != 0

D = RobotDynamics.DynamicsJacobian(n,m)
RD.jacobian!(RD.StaticReturn(), RD.ForwardAD(), model, D, y, z)
@test D.A == F[:,1:n]
@test D.B ≈ F[:,n .+ (1:m)]
RD.jacobian!(RD.StaticReturn(), RD.FiniteDifference(), model, D, y, z)
@test D.A ≈ F[:,1:n] atol=1e-6
@test D.B ≈ F[:,n .+ (1:m)] atol=1e-6

RD.jacobian!(RD.InPlace(), RD.ForwardAD(), model, D, y, z)
@test D.A == F[:,1:n]
@test D.B ≈ F[:,n .+ (1:m)]
RD.jacobian!(RD.InPlace(), RD.FiniteDifference(), model, D, y, z)
@test D.A ≈ F[:,1:n] atol=1e-6
@test D.B ≈ F[:,n .+ (1:m)] atol=1e-6

# Discretize
dmodel = RD.DiscretizedDynamics{RD.RK4}(model)
@test RD.discrete_dynamics(dmodel, x, u, t, dt) ≈ RD.discrete_dynamics(dmodel, z)

z_ = RD.KnotPoint(zero(x),zero(u),t,dt)
RD.propagate_dynamics!(RD.StaticReturn(), dmodel, z_, z)
@test RD.state(z_) ≈ RD.discrete_dynamics(dmodel, z)

F = zeros(n,n+m)
RD.jacobian!(RD.StaticReturn(), RD.ForwardAD(), dmodel, F, y, z)

# Error state
x0 = rand(model)[1]
@test RD.state_diff(model, x, x0) ≈ x - x0
@test RD.errstate_dim(model) == RD.state_dim(model)
G = zeros(n,n)
RD.state_diff_jacobian!(model, G, z)
@test G ≈ I(n)
G .= 0
RobotDynamics.state_diff_jacobian!(model, G, x)
@test G ≈ I(n)