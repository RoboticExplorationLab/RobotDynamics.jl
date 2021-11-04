
## Dynamics
abstract type AbstractModel <: AbstractFunction end

abstract type ContinuousDynamics <: AbstractModel end
@inline evaluate(model::ContinuousDynamics, z) =
    dynamics(model, state(z), control(z), time(z))
@inline evaluate!(model::ContinuousDynamics, ẋ, z) =
    dynamics!(model, ẋ, state(z), control(z), time(z))

abstract type DiscreteDynamics <: AbstractModel end
@inline evaluate(model::DiscreteDynamics, z) =
    discrete_dynamics(model, state(z), control(z), time(z), timestep(z))
@inline evaluate!(model::DiscreteDynamics, ẋ, z) =
    discrete_dynamics!(model, xn, state(z), control(z), time(z), timestep(z))

"Integration rule for approximating the continuous integrals for the equations of motion"
abstract type QuadratureRule end

"Integration rules of the form `x′ = f(x,u)`, where `x′` is the next state"
abstract type Explicit <: QuadratureRule end

@autodiff struct DiscretizedDynamics{L,Q} <: DiscreteDynamics
    continuous_dynamics::L
    integrator::Q
    function DiscretizedDynamics(
        continuous_dynamics::L,
        integrator::Q,
    ) where {L<:ContinuousDynamics,Q<:QuadratureRule}
        new{L, Q}(continuous_dynamics, Q)
    end
end
function DiscretizedDynamics{Q}(continuous_dynamics::L) where {L<:ContinuousDynamics} where Q
    n,m = size(continuous_dynamics)
    DiscretizedDynamics(continuous_dynamics, Q(n, m))
end

@inline integration(model::DiscretizedDynamics) = model.integrator

@inline evaluate(model::DiscretizedDynamics, z) = integrate(integration(model), model, z) 
@inline evaluate!(model::DiscretizedDynamics, xn, z) = integrate!(integration(model), model, xn, z) 
jacobian!(sig::FunctionSignature, ::UserDefined, model::DiscretizedDynamics, J, xn, z) = jacobian!(integration(model), sig, model, J, xn, z)