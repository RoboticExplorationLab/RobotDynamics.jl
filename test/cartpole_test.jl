using RobotDynamics
using Test
using StaticArrays

struct MyModel <: AbstractModel end
model = MyModel()
@test_throws ErrorException state_dim(model)
@test_throws ErrorException control_dim(model)

model = Cartpole()
x,u = zeros(model)
@test sum(x) == 0
@test sum(u) == 0
x,u = rand(model)
@test sum(x) != 0
@test sum(u) != 0
xdot = dynamics(model, x, u)
@test sum(x) != 0

x,u = zeros(Float32,model)
@test x isa SVector{4,Float32}
@test u isa SVector{1,Float32}

x,u = rand(Float32,model)
@test x isa SVector{4,Float32}
@test u isa SVector{1,Float32}

x,u = fill(model, 10.0)
@test x ≈ @SVector fill(10,4)
@test u ≈ @SVector fill(10,1)


n,m = size(model)
dt = 0.1
F = zeros(n,n+m)
z = KnotPoint(x,u,dt)
jacobian!(F, model, z)
@test sum(F) != 0

D = RobotDynamics.DynamicsJacobian(n,m)
jacobian!(D, model, z)
@test D.A == F[:,1:n]
@test D.B ≈ F[:,n .+ (1:m)]

@test discrete_dynamics(RK3, model, x, u, 0.0, dt) ≈
    discrete_dynamics(RK3, model, z)
@test discrete_dynamics(RK3, model, z) ≈ discrete_dynamics(model, z)

z_ = KnotPoint(zero(x),zero(m),dt)
RobotDynamics.propagate_dynamics(RK3, model, z_, z)
@test state(z_) ≈ discrete_dynamics(model, z)

F = zeros(n,n+m)
discrete_jacobian!(RK3, F, model, z)

tmp = [RobotDynamics.DynamicsJacobian(n,m) for k = 1:3]
jacobian!(RK3, D, model, z, tmp)
@test D.A ≈ F[1:n,1:n]
@test D.B ≈ F[1:n,n .+ (1:m)]
@test sum(F) != 0
@test F[1] == 1

# Error state
x0 = rand(model)[1]
@test RobotDynamics.state_diff(model, x, x0) ≈ x - x0
@test state_diff_size(model) == state_dim(model)
G = zeros(n,n)
RobotDynamics.state_diff_jacobian!(G, model, z)
@test G ≈ I(n)
G .= 0
RobotDynamics.state_diff_jacobian!(G, model, x)
@test G ≈ I(n)
@test RobotDynamics.state_diff_jacobian(model, x) ≈ G



# @btime discrete_jacobian!($RK3, $F, $model, $z)
# @btime jacobian!($RK3, $D, $model, $z, $tmp)
