abstract type DiscreteDynamics <: AbstractModel end
@inline evaluate(model::DiscreteDynamics, x, u, p) =
    discrete_dynamics(model, x, u, p.t, p.dt) 
@inline evaluate!(model::DiscreteDynamics, xn, x, u, p) =
    discrete_dynamics!(model, xn, x, u, p.t, p.dt)

discrete_dynamics(model::DiscreteDynamics, z::AbstractKnotPoint) =
    discrete_dynamics(model, state(z), control(z), time(z), timestep(z))
discrete_dynamics!(model::DiscreteDynamics, xn, z::AbstractKnotPoint) =
    discrete_dynamics!(model, xn, state(z), control(z), time(z), timestep(z))

discrete_dynamics(model::DiscreteDynamics, x, u, t, dt) =
    error("Discrete dynamics not defined yet.")
discrete_dynamics!(model::DiscreteDynamics, xn, x, u, t, dt) =
    error("In-place discrete dynamics not defined yet.")

jacobian!(model::DiscreteDynamics, J, y, x, u, p) = 
    jacobian!(model, J, y, x, u, p.t, p.dt)

dynamics_error(model::DiscreteDynamics, z2::AbstractKnotPoint, z1::AbstractKnotPoint) = discrete_dynamics(model, z1) - state(z2)
function dynamics_error!(model::DiscreteDynamics, y2, y1, z2::AbstractKnotPoint, z1::AbstractKnotPoint)
    discrete_dynamics!(model, y2, z1)
    y2 .-= state(z2)
    return nothing
end
# jacobian!(model::DiscreteDynamics, J2, J1, z2::KnotPoint, z1::KnotPoint)

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
        new{L,Q}(continuous_dynamics, Q)
    end
end
function DiscretizedDynamics{Q}(
    continuous_dynamics::L,
) where {L<:ContinuousDynamics} where {Q}
    n, m = size(continuous_dynamics)
    DiscretizedDynamics(continuous_dynamics, Q(n, m))
end

@inline integration(model::DiscretizedDynamics) = model.integrator
discrete_dynamics(model::DiscretizedDynamics, x, u, t, dt) =
    integrate(integration(model), model.continuous_dynamics, x, u, t, dt)
discrete_dynamics!(model::DiscretizedDynamics, xn, x, u, t, dt) =
    integrate!(integration(model), model.continuous_dynamics, xn, x, u, t, dt)

jacobian!(sig::FunctionSignature, ::UserDefined, model::DiscretizedDynamics, J, xn, z) =
    jacobian!(integration(model), sig, model.continuous_dynamics, J, xn, z)