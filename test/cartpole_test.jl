using RobotDynamics
using Test
using StaticArrays

struct Cartpole{T} <: AbstractModel
    mc::T
    mp::T
    l::T
    g::T
end

Cartpole() = Cartpole(1.0, 0.2, 0.5, 9.81)

function RobotDynamics.dynamics(model::Cartpole, x, u)
    mc = model.mc  # mass of the cart in kg (10)
    mp = model.mp   # mass of the pole (point mass at the end) in kg
    l = model.l   # length of the pole in m
    g = model.g  # gravity m/s^2

    q = x[ @SVector [1,2] ]
    qd = x[ @SVector [3,4] ]

    s = sin(q[2])
    c = cos(q[2])

    H = @SMatrix [mc+mp mp*l*c; mp*l*c mp*l^2]
    C = @SMatrix [0 -mp*qd[2]*l*s; 0 0]
    G = @SVector [0, mp*g*l*s]
    B = @SVector [1, 0]

    qdd = -H\(C*qd + G - B*u[1])
    return [qd; qdd]
end

RobotDynamics.state_dim(::Cartpole) = 4
RobotDynamics.control_dim(::Cartpole) = 1

model = Cartpole()
x,u = zeros(model)
@test sum(x) == 0
@test sum(u) == 0
x,u = rand(model)
@test sum(x) != 0
@test sum(u) != 0
xdot = dynamics(model, x, u)
@test sum(x) != 0

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

F = zeros(n,n+m)
discrete_jacobian!(RK3, F, model, z)

tmp = [RobotDynamics.DynamicsJacobian(n,m) for k = 1:3]
jacobian!(RK3, D, model, z, tmp)
@test D.A ≈ F[1:n,1:n]
@test D.B ≈ F[1:n,n .+ (1:m)]
@test sum(F) != 0
@test F[1] == 1

# @btime discrete_jacobian!($RK3, $F, $model, $z)
# @btime jacobian!($RK3, $D, $model, $z, $tmp)
