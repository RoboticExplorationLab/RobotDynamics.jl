## Dynamics
abstract type AbstractModel <: AbstractFunction end
output_dim(model::AbstractModel) = state_dim(model)

"""
    ContinuousDynamics

A dynamics model of the form:

```math
\\dot{x} = f(x,u)
```

where ``x`` is the `n`-dimensional state vector and ``u`` is the `m`-dimensional
control vector. The following methods should be defined on any model:

    state_dim(model)
    control_dim(model)

The output dimension `output_dim` is automatically defined to be equal to the 
state dimension. 

## Defining the dynamics
To define the dynamics using out-of-place (returning an SVector) evaluation, 
the user must define one of the following:

    dynamics(model, z::AbstractKnotPoint)
    dynamics(model, x, u, t)
    dynamics(model, x, u)

The user shouldn't assume that the inputs are static arrays, although for best 
computational performance these methods should be called by passing in static arrays.

To define the dynamics using in-place evaluation, the user must define one of the following:

    dynamics!(model, xdot, z::AbstractKnotPoint)
    dynamics!(model, xdot, x, u, t) 
    dynamics!(model, xdot, x, u) 

A user-defined Jacobian can be provided by defining one of the following methods

    jacobian!(model, J, xdot, z::AbstractKnotPoint)
    jacobian!(model, J, xdot, x, u, t)
    jacobian!(model, J, xdot, x, u)


## Non-Euclidean state vectors
By default, all elements of the state vector are assume to be in Euclidean space.
This assumption can be relaxed by defining a few extra functions. For non-Euclidean 
state vectors, the Euclidean difference between two states, or the error state, is no 
longer computed using pure subtraction. We can define a custom function for taking the 
differences between 2 states vectors based on the type of the state vector. 

We can query the type of the state vector using [`statevectortype(model)`](@ref), 
which returns a [`StatVectorType`](@ref) trait (by default, [`EuclideanState`](@ref)).
After defining a new `StateVectorType` and the following methods (described in more 
detail in the documentation for [`StateVectorType`](@ref)):

    state_diff!
    errstate_dim
    state_diff_jacobian!
    ∇state_diff_jacobian!
    
We can use dynamics with non-Euclidean state vectors. The most commonly encounter 
case of this is when using 3D rotations. Support for states with 3D rotations is 
provided by the [`RotationState`](@ref) `StateVectorType`.
"""
abstract type ContinuousDynamics <: AbstractModel end

"""
    dynamics(model, z::AbstractKnotPoint)
    dynamics(model, x, u, t)
    dynamics(model, x, u)

Evaluate the continuous time dynamics, returning the output ``\\dot{x}``. For best 
performance, the output should usually be a `StaticArrays.SVector`. This method is 
called when using the `StaticReturn` [`FunctionSignature`](@ref).

Calling `evaluate` on a [`ContinuousDynamics`](@ref) model will call this function.
"""
@inline dynamics(model::ContinuousDynamics, z::AbstractKnotPoint) =
    dynamics(model, state(z), control(z), time(z))
@inline dynamics(model::ContinuousDynamics, x, u, t) = dynamics(model, x, u)

"""
    dynamics!(model, xdot, z::AbstractKnotPoint)
    dynamics!(model, xdot, x, u, t)
    dynamics!(model, xdot, x, u)

Evaluate the continuous time dynamics, storing the output in `xdot`. 
This method is called when using the `InPlace` [`FunctionSignature`](@ref).

Calling `evaluate!` on a [`ContinuousDynamics`](@ref) model will call this function.
"""
@inline dynamics!(model::ContinuousDynamics, xdot, z::AbstractKnotPoint) =
    dynamics!(model, xdot, state(z), control(z), time(z))
@inline dynamics!(model::ContinuousDynamics, ẋ, x, u, t) = dynamics!(model, ẋ, x, u)

"""
    dynamics!(sig, model, xdot, z::AbstractKnotPoint)

Evaluate the continuous time dynamics, storing the output in `xdot`, using the 
[`FunctionSignature`](@ref) `sig` to determine which method to call.
"""
dynamics!(::InPlace, model::ContinuousDynamics, xdot, z::AbstractKnotPoint) = 
    dynamics!(model, xdot, z)
dynamics!(::StaticReturn, model::ContinuousDynamics, xdot, z::AbstractKnotPoint) = 
    xdot .= dynamics!(model, z)

@inline evaluate(model::ContinuousDynamics, x, u, p) =
    dynamics(model, x, u, p.t) 
@inline evaluate!(model::ContinuousDynamics, ẋ, x, u, p) =
    dynamics!(model, ẋ, x, u, p.t) 

@inline jacobian!(::FunctionSignature, ::UserDefined, model::ContinuousDynamics, J, ẋ, z) =
    jacobian!(model, J, ẋ, state(z), control(z), time(z))

