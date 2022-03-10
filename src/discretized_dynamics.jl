"""
    QuadratureRule

Integration rule for approximating the continuous integrals for the equations of motion.
Currently divided into two classes of integrators:
* [`Explicit`](@ref)
* [`Implicit`](@ref)

## Interface
All integrators need to have a constructor of the form:

    Q(n, m)

where `n` and `m` are the state and control dimensions of the dynamical system.
"""
abstract type QuadratureRule end

"""
    Explicit <: QuadratureRule

Integration rules of the form 
```math
x_{k+1} = f(x_k,u_k)
```.

## Interface
All explicit integrators need to define the following methods:

    integrate(::Q, model, x, u, t, h)
    integrate!(::Q, model, xn, x, u, t, h)
    jacobian!(::Q, sig, model, J, xn, x, u, t, h)

where `h` is the time step (i.e. `dt`). The Jacobian `J` should have size `(n, n + m)`.
The `sig` is a [`FunctionSignature`](@ref) that can be used to dispatch the Jacobian on 
the desired function signature. For example, when using `StaticReturn` intermediate 
cache variables aren't usually needed, but are when using `InPlace` methods. These cache
variables can be stored in the integrator itself (see the [`ADVector`](@ref) type provided
by this package).

!!! note
    The `Q` in these methods should be replaced by your explicit integrator, 
    e.g. `MyRungeKutta`.

"""
abstract type Explicit <: QuadratureRule end

"""
    Implicit <: QuadratureRule

Integration rules of the form 
```math
f(x_{k+1},u_{k+1},x_k,u_k) = 0
```.

## Interface
All implicit integrators need to define the following methods:

    dynamics_error(::Q, model, z2, z1)
    dynamics_error!(::Q, model, y2, y1, z2, z1)
    dynamics_error_jacobian!(::Q, sig, model, J2, J1, y2, y1, z2, z1)

where `model` is a [`ContinuousDynamics`](@ref) model. For the in place method, 
the output should be stored in the `y2` vector. For the Jacobian method, `J2` 
holds the Jacobian with respect to the state and control of `z2` (the knot point 
at the next time step) and `J1` holds the Jacobian with respect to the state 
and control of `z1` (the knot point at the current time step).
"""
abstract type Implicit <: QuadratureRule end

# Create an integrator from a dynamics model
(::Type{Q})(model::AbstractModel) where {Q<:QuadratureRule} = Q(state_dim(model), control_dim(model))

"""
    DiscretizedDynamics

Represents a [`DiscreteDynamics`](@ref) model formed by integrating a continuous 
dynamics model. It is essentially a [`ContinuousDynamics`](@ref) paired with a 
[`QuadratureRule`](@ref) that defines how to use the [`dynamics!`](@ref) function 
to get a [`discrete_dynamics!`](@ref) function. 

## Constructor
A `DiscretedDynamics` type can be created using either of the following signatures:

    DiscretizedDynamics(dynamics::ContinuousDynamics, Q::QuadratureRule)
    DiscretizedDynamics{Q}(dynamics::ContinuousDynamics) where Q <: QuadratureRule

In the second case, the integrator is constructed by calling 
`Q(state_dim(dynamics), control_dim(dynamics))`.

## Usage
A `DiscretizedDynamics` model is used just like any other [`DiscreteDynamics`](@ref) model.
The state, control, and error state dimensions are all taken from the 
continuous time dynamics model, and the [`StateVectorType`](@ref) and 
[`FunctionInputs`](@ref) traits are inherited from the continuous time dynamics.

The `default_diffmethod`, however, is set to `ForwardAD`, since our benchmarks show that 
it is usually faster to make a single call to `ForwardDiff.jacobian!` than using the chain 
rule and multiple calls to query the Jacobian of the continuous dynamics at multiple points.
If the continuous model has a `UserDefined` Jacobian method, calling `jacobian!` on 
a `DiscretizedDynamics` model will use the chain rule with analytical Jacobians of 
the integrator. If the user wants to use a combination of ForwardDiff (or any other 
differentiation method, for that matter) with the integrator, they are free to define their 
own [`DiffMethod`](@ref) to dispatch on.
"""
@autodiff struct DiscretizedDynamics{L,Q} <: DiscreteDynamics
    continuous_dynamics::L
    integrator::Q
    function DiscretizedDynamics(
        continuous_dynamics::L,
        integrator::Q,
    ) where {L<:ContinuousDynamics,Q<:QuadratureRule}
        new{L,Q}(continuous_dynamics, integrator)
    end
end
function DiscretizedDynamics{Q}(
    continuous_dynamics::L,
) where {L<:ContinuousDynamics} where {Q}
    n, m = dims(continuous_dynamics)
    DiscretizedDynamics(continuous_dynamics, Q(n, m))
end

const ImplicitDynamicsModel{L,Q} = DiscretizedDynamics{L,Q} where {L,Q<:Implicit}

for method in (:state_dim, :control_dim, :output_dim, :errstate_dim, :statevectortype, 
               :functioninputs)
    @eval $method(model::DiscretizedDynamics) = $method(model.continuous_dynamics)
end
default_diffmethod(::DiscretizedDynamics) = ForwardAD()

@inline integration(model::DiscretizedDynamics) = model.integrator
discrete_dynamics(model::DiscretizedDynamics, x, u, t, dt) =
    integrate(integration(model), model.continuous_dynamics, x, u, t, dt)
discrete_dynamics!(model::DiscretizedDynamics, xn, x, u, t, dt) =
    integrate!(integration(model), model.continuous_dynamics, xn, x, u, t, dt)

jacobian!(sig::FunctionSignature, ::UserDefined, model::DiscretizedDynamics, J, xn, z) =
    jacobian!(
        integration(model),
        sig,
        model.continuous_dynamics,
        J,
        xn,
        state(z),
        control(z),
        time(z),
        timestep(z),
    )

########################################
# Implicit Dynamics
########################################

# TODO: [#19] overwrite `evaluate` to solve for the next state using Newton
# TODO: [#19] overwrite `jacobian` to provide A,B using implicit function theorem
function evaluate!(model::ImplicitDynamicsModel, xn, z::AbstractKnotPoint)
    # TODO: get these from some options
    newton_iters = 10
    tol = 1e-6

    z1 = z
    n,m = dims(z)
    integrator = integration(model)
    J2 = integrator.J2
    J1 = integrator.J1
    r = integrator.y2
    dx = integrator.y1
    z2 = integrator.z2
    ipiv = integrator.ipiv
    copyto!(z2, z1)
    for iter = 1:newton_iters
        # Set the guess for the next state
        setstate!(z2, xn)

        # Calculate the residual
        dynamics_error!(integrator, model.continuous_dynamics, r, dx, z2, z1)

        if norm(r) < tol
            break
        end

        # Calculate the Jacobian wrt x2
        dynamics_error_jacobian!(integrator, InPlace(), model.continuous_dynamics, J2, J1, r, dx, z2, z1)
        A = @view J2[:, 1:n]

        # Get the step
        dx .= r
        F = lu!(A, ipiv)
        ldiv!(F, dx)

        # Apply the step
        xn .-= dx
    end
end

dynamics_error(
    model::ImplicitDynamicsModel,
    z2::AbstractKnotPoint,
    z1::AbstractKnotPoint,
) = dynamics_error(integration(model), model.continuous_dynamics, z2, z1)

dynamics_error!(
    model::ImplicitDynamicsModel,
    y2,
    y1,
    z2::AbstractKnotPoint,
    z1::AbstractKnotPoint,
) = dynamics_error!(integration(model), model.continuous_dynamics, y2, y1, z2, z1)

dynamics_error_jacobian!(
    sig::FunctionSignature,
    ::UserDefined,
    model::ImplicitDynamicsModel,
    J2,
    J1,
    y2,
    y1,
    z2::AbstractKnotPoint,
    z1::AbstractKnotPoint,
) = dynamics_error_jacobian!(
    integration(model),
    sig,
    model.continuous_dynamics,
    J2,
    J1,
    y2,
    y1,
    z2,
    z1,
)
