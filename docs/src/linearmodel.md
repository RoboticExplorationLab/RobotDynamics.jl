```@meta
CurrentModule = RobotDynamics
```

# Linear Models
RobotDynamics supports the easy construction of linear models. By defining a linear model, the relevant dynamics
and jacobian functions are predefined for you. This can result in signicant speed ups compared to a naive 
specification of a standard continuous model. 

```@docs
LinearModel
LinearizedModel
```

# Linearizing and Discretizing a Model (experimental)
Many systems with complicated nonlinear dynamics can be simplified by linearizing them about a fixed point
or a trajectory. This can allow the use of specialized and faster trajectory optimization methods for these
linear systems. The functions that RobotDynamics provides also discretize the system. 

```@docs
linearize_and_discretize!
discretize!
update_trajectory!
```

# Example
Take for example the cartpole, which can be readily linearized about it's stable point. We can use the 
`LinearizedModel` to easily find the linearized system.

```julia
using RobotDynamics
import RobotDynamics: dynamics  # the dynamics function must be imported
using StaticArrays

const RD = RobotDynamics

struct Cartpole{T} <: AbstractModel
    mc::T  # mass of the cart
    mp::T  # mass of the pole
    l::T   # length of the pole
    g::T   # gravity
end

function dynamics(model::Cartpole, x, u)
    mc = model.mc   # mass of the cart in kg (10)
    mp = model.mp   # mass of the pole j(point mass at the end) in kg
    l = model.l     # length of the pole in m
    g = model.g     # gravity m/s^2

    q  = x[SA[1,2]]
    qd = x[SA[3,4]]

    s = sin(q[2])
    c = cos(q[2])

    H = SA[mc+mp mp*l*c; mp*l*c mp*l^2]
    C = SA[0 -mp*qd[2]*l*s; 0 0]
    G = SA[0, mp*g*l*s]
    B = SA[1, 0]

    qdd = -H\(C*qd + G - B*u[1])
    return [qd; qdd]
end

RD.state_dim(::Cartpole) = 4
RD.control_dim(::Cartpole) = 1

nonlinear_model = Cartpole(1.0, 1.0, 1.0, 9.81)
n = state_dim(nonlinear_model)
m = control_dim(nonlinear_model)

# stationary point for the cartpole around which to linearize
x̄ = @SVector [0., π, 0., 0.]
ū = @SVector [0.0]
dt = 0.01
knot_point = KnotPoint(x̄, ū, dt)

# creates a new LinearizedModel around stationary point
linear_model = RD.LinearizedModel(nonlinear_model, knot_point, Exponential)

δx = @SVector zeros(n)
δu = @SVector zeros(m)

# outputs linearized dynamics!
δxₖ₊₁ = discrete_dynamics(PassThrough, linear_model, δx, δu, 0.0, dt) 

@assert δxₖ₊₁ ≈ zeros(n)

F = RD.DynamicsJacobian(n,m)
discrete_jacobian!(PassThrough, F, linear_model, knot_point)

@show A = RD.get_A(F)
@show B = RD.get_B(F)

```
