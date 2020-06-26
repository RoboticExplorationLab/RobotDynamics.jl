# [2. Evaluating the Dynamics](@id model_section)
```@meta
CurrentModule = RobotDynamics
```

```@contents
Pages = ["dynamics_evaluation.md"]
```

## Overview
Once we have a model defined as detailed on the previous page, we can query both the
continuous and discrete dynamics, and have access to some additional, useful methods.

## Querying the Continuous Dynamics
We can evaluate the continuous dynamics using one of the following methods:

```julia
ẋ = dynamics(model, z)
ẋ = dynamics(model, x, u)
ẋ = dynamics(model, x, u, t)
```
where `z` is an [`AbstractKnotPoint`](@ref), `x` is the `n`-dimensional state vector,
`u` is the `m`-dimensional control vector, and `t` is the positive scalar independent variable,
typically time. For best performance, `x` and `u` should be `SVector`s.

We can evaluate the continuous time dynamics Jacobian using the method
```julia
jacobian!(∇f, model, z)
```
where `∇f` is an `n × (n+m)` matrix. Note that the Jacobian methods require an `AbstractKnotPoint`,
since this eliminates unnecessary concatenation and subsequent memory allocations when using
ForwardDiff.

### The `DynamicsJacobian` type
While the dynamics Jacobian `∇f` can be any `AbstractMatrix`, RobotDynamics provides the
`DynamicsJacobian` type that has some convenient constructors and provides access to the
individual partial derivatives:

```@docs
DynamicsJacobian
```

## Querying Discrete Dynamics
The discrete dynamics can be evaluated using methods analogous to those used to evaluate
the continuous dynamics, except we now need to specify the integration method and the time step.

```julia
x′ = discrete_dynamics(::Type{Q}, model, z)
x′ = discrete_dynamics(::Type{Q}, model, x, u, t, dt)
```
where `Q` is a [`QuadratureRule`](@ref). See [Discretization](@ref) for more information on
the integration methods defined in RobotDynamics.

When evaluating discrete dynamics, one can also use the `propagate_dynamics` method that
updates the state of the next `KnotPoint`:
```@docs
propagate_dynamics
```

The discrete dynamics Jacobian is similarly evaluated using
```julia
discrete_jacobian!(::Type{Q}, ∇f, model, z)
```

If the integration method is not passed in as the first argument, the default integration
method [`RK3`](@ref) will be used.


## Other Methods
All `AbstractModel`s provide a few functions for generating state and control vectors directly
from the model:
```julia
x,u = zeros(model)
x,u = rand(model)
x,u = fill(model, value)
```

The `Base.size` method is also overloaded as a shortcut for returning `state_dim(model)`
and `control_dim(model)` as a tuple:
```julia
n,m = size(model)
```
