"Integration rule for approximating the continuous integrals for the equations of motion"
abstract type QuadratureRule end

"Integration rules of the form `x′ = f(x,u)`, where `x′` is the next state"
abstract type Explicit <: QuadratureRule end

"Integration rules of the form x′ = f(x,u,x′,u′), where x′,u′ are the states and controls at the next time step."
abstract type Implicit <: QuadratureRule end

(::Type{Q})(model::AbstractModel) where {Q<:QuadratureRule} = Q(state_dim(model), control_dim(model))

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

const ImplicitDynamicsModel{L,Q} = DiscretizedDynamics{L,Q} where {L,Q<:Implicit}

@inline state_dim(model::DiscretizedDynamics) = state_dim(model.continuous_dynamics)
@inline control_dim(model::DiscretizedDynamics) = control_dim(model.continuous_dynamics)
@inline errstate_dim(model::DiscretizedDynamics) = errstate_dim(model.continuous_dynamics)
@inline state_diff(model::DiscretizedDynamics, x, x0) = state_diff(model.continuous_dynamics, x, x0)
statevectortype(::Type{<:DiscretizedDynamics{L}}) where L = statevectortype(L)

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

# TODO: overwrite `evaluate` to solve for the next state using Newton
# TODO: overwrite `jacobian` to provide A,B using implicit function theorem

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
