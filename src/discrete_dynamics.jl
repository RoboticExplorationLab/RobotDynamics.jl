"""
    DiscreteDynamics

A dynamics model of the form

```math
x_{k+1} = f(x_k, u_k)
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

    discrete_dynamics(model, z::AbstractKnotPoint)
    discrete_dynamics(model, x, u, t, dt)

where `t` is the current time and `dt` is the time step between ``x_k`` and ``x_{k+1}``.

The user shouldn't assume that the inputs are static arrays, although for best 
computational performance these methods should be called by passing in static arrays.

To define the dynamics using in-place evaluation, the user must define one of the following:

    discrete_dynamics!(model, xn, z::AbstractKnotPoint)
    discrete_dynamics!(model, xn, x, u, t, dt) 

Either of these function can be called by dispatching on the [`FunctionSignature`](@ref)
using:

    discrete_dynamics!(sig, model, xn, z::AbstractKnotPoint)

## Discretizing continuous dynamics
This type is the abstract type representing all discrete dynamics systems, whether 
or not they are approximations of continuous systems. Users can choose to implement 
their dynamics directly in discrete time using this interface, or can use 
[`DiscretizedDynamics`](@ref) to discretize a continuous time system.

## Implicit dynamics functions
Sometimes continuous time systems are approximated using implicit integrators of the 
form 

```math
d(x_{k+1}, u_{k+1}, x_k, u_k) = 0
```

To evaluate these types of systems, use the [`dynamics_error`](@ref) and 
[`dynamics_error!`](@ref) methods, which will return the output of `f` for 
two consecutive knot points. The default functions such as `discrete_dynamics` or
`jacobian!` are not defined for these types of systems.

## Simulating the dynamics
Use [`propagate_dynamics!`](@ref) to conveniently save the next state directly into 
the state vector of another [`KnotPoint`](@ref).
"""
abstract type DiscreteDynamics <: AbstractModel end
@inline evaluate(model::DiscreteDynamics, z::AbstractKnotPoint) =
    discrete_dynamics(model, z::AbstractKnotPoint)
@inline evaluate!(model::DiscreteDynamics, xn, z::AbstractKnotPoint) =
    discrete_dynamics!(model, xn, z::AbstractKnotPoint)

"""
    discrete_dynamics(model, z::AbstractKnotPoint)
    discrete_dynamics(model, x, u, t, dt)

Evaluate the discrete time dynamics, returning the output ``x_{k+1}``. For best 
performance, the output should be a `StaticArrays.SVector`. This method is called 
when using the `StaticReturn` [`FunctionSignature`](@ref).

Calling `evaluate` on a [`DiscreteDynamics`](@ref) model will call this function.
"""
discrete_dynamics(model::DiscreteDynamics, z::AbstractKnotPoint) =
    discrete_dynamics(model, state(z), control(z), time(z), timestep(z))
discrete_dynamics(model::DiscreteDynamics, x, u, t, dt) =
    error("Discrete dynamics not defined yet.")

"""
    discrete_dynamics!(model, xn, z::AbstractKnotPoint)
    discrete_dynamics!(model, xn, x, u, t, dt)

Evaluate the discrete time dynamics, storing the output in `xn`. 
This method is called when using the `InPlace` [`FunctionSignature`](@ref).

Calling `evaluate!` on a [`DiscreteDynamics`](@ref) model will call this function.
"""
discrete_dynamics!(model::DiscreteDynamics, xn, z::AbstractKnotPoint) =
    discrete_dynamics!(model, xn, state(z), control(z), time(z), timestep(z))
discrete_dynamics!(model::DiscreteDynamics, xn, x, u, t, dt) =
    error("In-place discrete dynamics not defined yet.")

"""
    discrete_dynamics!(sig, xn, z::AbstractKnotPoint)

Evaluate the discrete dynamics function, storing the output in `xn`, using the 
[`FunctionSignature`](@ref) `sig` to determine which method to call.
"""
discrete_dynamics!(::InPlace, model::DiscreteDynamics, xn, z::AbstractKnotPoint) = 
    discrete_dynamics!(model, xn, z)
discrete_dynamics!(::StaticReturn, model::DiscreteDynamics, xn, z::AbstractKnotPoint) = 
    xn .= discrete_dynamics(model, z)


jacobian!(model::DiscreteDynamics, J, y, x, u, p) = jacobian!(model, J, y, x, u, p.t, p.dt)
jacobian!(model::DiscreteDynamics, J, y, x, u, t, dt) =
    error("User-defined discrete dynamics Jacobian not defined.")

"""
    dynamics_error(model, z2, z1)

Evaluate the dynamics error between two knot points for a [`DiscreteDynamics`](@ref) 
model. In general, this function takes the form:

```math
d(x_{k+1}, u_{k+1}, x_k, u_k) = 0
```

For explicit integration methods of the form ``x_{k+1} = f(x_k, u_k)``, this is just
```math
d(x_k, u_k) - x_{k+1} = 0
```

This is the method that should be used with implicit integration methods.

This form is the out-of-place form that should, in general, return a 
`StaticVectors.SVector`. See [`dynamics_error!`](@ref) for the in-place method.

The Jacobian of this function is evaluated by calling [`dynamics_error_jacobian!`](@ref).
"""
dynamics_error(model::DiscreteDynamics, z2::AbstractKnotPoint, z1::AbstractKnotPoint) =
    discrete_dynamics(model, z1) - state(z2)

"""
    dynamics_error!(model, y2, y1, z2, z1)

Evaluate the dynamics error between two knot points for a [`DiscreteDynamics`](@ref) model.
The output is stored in `y2`, and `y1` is provided as a extra input that can be used 
to store temporary results. Any specific usage of this variable is left to the user.

See [`dynamics_error`](@ref) for more details on this function.
"""
function dynamics_error!(
    model::DiscreteDynamics,
    y2,
    y1,
    z2::AbstractKnotPoint,
    z1::AbstractKnotPoint,
)
    discrete_dynamics!(model, y2, z1)
    y2 .-= state(z2)
    return nothing
end

"""
    dynamics_error_jacobian!(sig, diff, model, J2, J1, y2, y1, z2, z1)

Evaluate the Jacobian of [`dynamics_error`](@ref). The derivative with respect to 
the first knotpoint `z1` should be stored in `J1`, and the derivative with respect to 
the second knotpoint `z2` should be stored in `J2`. The variables `y2` and `y1` should be 
vectors of the size of the output dimension (usually the state dimension) and are provided 
as extra cache variables whose usage is left to be determined by the user. For example, 
a user could choose to evaluate the dynamics error and store the result in one of these 
variables to evaluate the error and it's Jacobian at the same time.

Both `J2` and `J1` must have dimensions `(p, n + m)` where `n`, `m`, and `p` are the output 
of `state_dim`, `control_dim`, and `output_dim`. Usually `p = n`.

Note that for explicit integration methods of the form `x_{k+1} = f(x_k, u_k)` `J2` should 
be equal to `[-I(n) zeros(n,m)]`.

## Implementing on a custom type
To implement this function on a new [`DiscreteDynamics`](@ref) model, define methods 
for the [`FunctionSignature`](@ref) and [`DiffMethod`](@ref) of your choice. These are 
automatically defined when using [`@autodiff`](@ref) on a [`DiscreteDynamics`](@ref) model.
To implement the `UserDefined` [`DiffMethod`](@ref), implement the following method:

    dynamics_error_jacobian!(model, J2, J1, y2, y1, 
                             z2::AbstractKnotPoint, z1::AbstractKnotPoint)

"""
dynamics_error_jacobian!(
    ::FunctionSignature,
    ::UserDefined,
    model::DiscreteDynamics,
    J2,
    J1,
    y2,
    y1,
    z2::AbstractKnotPoint,
    z1::AbstractKnotPoint,
) = dynamics_error_jacobian!(model, J2, J1, y2, y1, z2, z1)

dynamics_error_jacobian!(
    model::DiscreteDynamics,
    J2,
    J1,
    y2,
    y1,
    z2::AbstractKnotPoint,
    z1::AbstractKnotPoint,
) = error("User-defined dynamics error Jacobian not defined for $(typeof(model)).")

"""
    propagate_dynamics!(sig, model, z2, z1)

Save the output of either [`discrete_dynamics`](@ref) or [`discrete_dynamics!`](@ref)
evaluated using `z1` into the state vector of `z2`. Useful for simulating discrete systems 
forward in time.
"""
function propagate_dynamics!(
    ::InPlace,
    model::DiscreteDynamics,
    z2::AbstractKnotPoint,
    z1::AbstractKnotPoint,
)
    discrete_dynamics!(model, state(z2), z1)
    return nothing
end

function propagate_dynamics!(
    ::StaticReturn,
    model::DiscreteDynamics,
    z2::AbstractKnotPoint,
    z1::AbstractKnotPoint,
)
    setstate!(z2, discrete_dynamics(model, z1))
    return nothing
end