# [1. Setting up a Dynamics Model](@id model_section)
```@meta
CurrentModule = RobotDynamics
```

```@contents
Pages = ["models.md"]
```
## Overview
The Model type holds information about the dynamics of the system.
All dynamics are assumed to be state-space models of the system of the form
``\dot{x} = f(x,u)`` where ``\dot{x}`` is the state derivative,
``x`` an ``n``-dimensional state vector, and ``u`` in an ``m``-dimensional control input vector.
The function ``f`` can be any nonlinear function.

Many numerical methods require discrete dynamics of the form
``x_{k+1} = f(x_k, u_k)``, where ``k`` is the time step.
There many methods of performing this discretization, and RobotDynamics.jl offers several of
the most common methods. See [Model Discretization](@ref) section for more information on
discretizing dynamics, as well as how to define custom integration methods.


## Creating a New Model
To create a new model of a dynamical system, you need to define a new type that inherits from `AbstractModel`. You will need to then define only a few methods on your type. Let's say we want to create a model of the canonical cartpole. We start by defining our type:
```julia
struct Cartpole{T} <: AbstractModel
    mc::T  # mass of the cart
    mp::T  # mass of the pole
    l::T   # length of the pole
    g::T   # gravity
end
```
It's often convenient to store any model parameters inside the new type (make sure they're concrete types!). If you need to store vectors or matrices, we highly recommend using StaticArrays, which are extremely fast and avoid memory allocations. For models with lots of parameters, we recommend [Parameters.jl](https://github.com/mauro3/Parameters.jl) that makes it easy to specify default parameters.

We now just need to define two functions to complete the interface
```julia
import RobotDynamics: dynamics  # the dynamics function must be imported

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
```

And voila! we have a new model.

We now have a few methods automatically available to us:
```julia
dynamics(model, z)
jacobian!(∇f, model, z)
```

We can also use `RD.dims(model)` to get `(n,m,n)`, `rand(model)` to get a tuple of randomly-sampled state and
control vectors, or `zeros(model)` to get 0-vectors of the state and control.

### Analytical Jacobians
Instead of relying on ForwardDiff to generate our dynamics Jacobian, we can instead overwrite
the method ourselves by defining the function:

```
jacobian!(∇f, model::Cartpole, z::AbstractKnotPoint)
```
where `∇f` is a `n × (n+m)` matrix and `z` is an [`AbstractKnotPoint`](@ref).

!!! warning
    By default, RobotDynamics will NOT use the analytical continuous Jacobian when computing
    the discrete Jacobian, since our benchmarks have shown it is typically faster to let
    ForwardDiff compute the Jacobian directly on the discrete dynamics function, thereby
    avoiding multiple calls to `jacobian!`.


## Time-varying systems
RobotDynamics.jl also offers support for time-varying systems. Let's say
for some reason the mass of our cartpole is decreasing linearly with time. We can model this
with a slight modification to the dynamics function signature:

```julia
import RobotDynamics: dynamics

struct CartpoleTimeVarying{T} <: AbstractModel
    mc::T  # initial mass of the cart
    mp::T  # mass of the pole
    l::T   # length of the pole
    g::T   # gravity
end

function dynamics(model::CartpoleTimeVarying, x, u, t)  # note extra time parameter
    mc = model.mc   # mass of the cart in kg (10)
    mp = model.mp   # mass of the pole (point mass at the end) in kg
    l = model.l     # length of the pole in m
    g = model.g     # gravity m/s^2

    # Change the mass of the cart with time
    mc = mc - 0.01*t

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

RobotDynamics.state_dim(::CartpoleTimeVarying) = 4
RobotDynamics.control_dim(::CartpoleTimeVarying) = 1
```

## Discrete Dynamical Systems
Most models are assumed to be continuous in nature, and require some integration scheme
(such as a Runge-Kutta method) to convert to discrete-time dynamics. However, some systems
are naturally discrete or perhaps the user has a custom integration method already applied
to their system. Instead of defining the continuous dynamics function, we can directly
define the discrete dynamics instead with the predefined integration type [`PassThrough`](@ref):

```julia
# Define the discrete dynamics function
function RobotDynamics.discrete_dynamics(::Type{PassThrough}, model::Cartpole,
        x::StaticVector, u::StaticVector, t, dt)

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
    xdot = [qd; qdd]
    return x + xdot * dt  # simple Euler integration
end
```

## Models with 3D Rotations
RobotDynamics.jl offers support for models with non-Euclidean state
vectors, such as 3D rotations, which live in ``SO(3)`` instead of ``\mathbb{R}^4`` (quaternions)
or ``\mathbb{R}^3`` (Euler angles, Modified Rodrigues Parameters, etc.). See [`RigidBody`](@ref)
section for more details.
