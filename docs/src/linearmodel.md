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
linearize_and_discretize
linearize_and_discretize!
discretize!
```

# Example
Take for example the cartpole, which can be readily linearized about it's stable point. We can use the 
`linearize_and_discretize` method to easily find the linearized system.

```julia
import RobotDynamics: dynamics  # the dynamics function must be imported

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

RobotDynamics.state_dim(::Cartpole) = 4
RobotDynamics.control_dim(::Cartpole) = 1

nonlinear_model = Cartpole(1.0, 1.0, 1.0, 9.81)
n = state_dim(nonlinear_model)
m = control_dim(nonlinear_model)

x̄ = @SVector [0., π, 0., 0.]
ū = @SVector [0.0]
dt = 0.01
knot_point = KnotPoint(x̄, ū, dt)

# puts linearized and discretized model into linear_model
linear_model = linearize_and_discretize(Exponential, nonlinear_model, knot_point)

# outputs linearized dynamics!
discrete_dynamics(PassThrough, linear_model, x̄, ū) 
# discrete_jacobian! is also defined
```
