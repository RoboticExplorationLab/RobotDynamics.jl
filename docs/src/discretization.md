```@meta
CurrentModule =
```

# Discretization
This page gives details on the methods for evaluating discretized dynamics, as well as instructions
on how to define a custom integration method.

## Model Discretization
With a model defined, we can compute the discrete dynamics and discrete dynamics Jacobians for an Implicit
integration rule with the following methods

```@docs
discrete_dynamics
discrete_jacobian!
```

## Integration Schemes
RobotDynamics.jl has already defined a handful of integration schemes for computing discrete dynamics.
The integration schemes are specified as abstract types, so that methods can efficiently dispatch
based on the integration scheme selected. Here is the current set of implemented types:
* [`QuadratureRule`](@ref)
    * [`Implicit`](@ref)
        * [`RK2`](@ref)
        * [`RK3`](@ref)
        * [`RK4`](@ref)
    * [`Explicit`](@ref)
        * [`HermiteSimpson`](@ref)

```@docs
QuadratureRule
RobotDynamics.Explicit
RK2
RK3
RK4
RobotDynamics.Implicit
HermiteSimpson
```

## Defining a New Integration Scheme

### Explicit Methods
Explicit integration schemes are understandably simpler, since the output is not a function of
itself, as is the case with implict schemes. As such, as a minimum, the user only needs to define
the following method for a new rule `MyQ`:

```julia
abstract type MyQ <: RobotDynamics.Explicit end
x′ = discrete_dynamics(::Type{MyQ}, model::AbstractModel, x, u, dt)
```

### Implicit Methods (experimental)
Incorporating implicit integration methods is still under development (great option for
    someone looking to contribute!).
