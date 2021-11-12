
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
    n, m = size(continuous_dynamics)
    DiscretizedDynamics(continuous_dynamics, Q(n, m))
end

state_dim(model::DiscretizedDynamics) = state_dim(model.continuous_dynamics)
control_dim(model::DiscretizedDynamics) = control_dim(model.continuous_dynamics)

@inline integration(model::DiscretizedDynamics) = model.integrator
discrete_dynamics(model::DiscretizedDynamics, x, u, t, dt) =
    integrate(integration(model), model.continuous_dynamics, x, u, t, dt)
discrete_dynamics!(model::DiscretizedDynamics, xn, x, u, t, dt) =
    integrate!(integration(model), model.continuous_dynamics, xn, x, u, t, dt)

# jacobian!(model::DiscretizedDynamics, J0, xdot, x, u, t, h) = 
#     jacobian!(integration(model), sig, model.continuous_dynamics, J, xn, z)

jacobian!(sig::FunctionSignature, ::UserDefined, model::DiscretizedDynamics, J, xn, z) =
    jacobian!(integration(model), sig, model.continuous_dynamics, J, xn, state(z), control(z), time(z), timestep(z))