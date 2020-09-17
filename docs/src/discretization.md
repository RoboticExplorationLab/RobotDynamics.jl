```@meta
CurrentModule = RobotDynamics
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
    * [`Explicit`](@ref)
        * [`RK2`](@ref)
        * [`RK3`](@ref)
        * [`RK4`](@ref)
        * [`Exponential`](@ref)
        * [`PassThrough`](@ref)
    * [`Implicit`](@ref)
        * [`HermiteSimpson`](@ref)


```@docs
QuadratureRule
RobotDynamics.Explicit
RK2
RK3
RK4
Exponential
PassThrough
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
x′ = discrete_dynamics(::Type{MyQ}, model::AbstractModel, x, u, t, dt)
```
which will make calls to the continuous-time dynamics function `dynamics(model, x, u, t)`.

Below is an example of the default integration method [`RK3`](@ref), a third-order Runge-Kutta method:
```julia
function discrete_dynamics(::Type{RK3}, model::AbstractModel,
		x::StaticVector, u::StaticVector, t, dt)
    k1 = dynamics(model, x,             u, t       )*dt;
    k2 = dynamics(model, x + k1/2,      u, t + dt/2)*dt;
    k3 = dynamics(model, x - k1 + 2*k2, u, t + dt  )*dt;
    x + (k1 + 4*k2 + k3)/6
end
```

### Implicit Methods (experimental)
Incorporating implicit integration methods is still under development (great option for
    someone looking to contribute!).
