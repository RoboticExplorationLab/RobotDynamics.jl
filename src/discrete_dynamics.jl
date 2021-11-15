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

jacobian!(model::DiscreteDynamics, J, y, x, u, p) = jacobian!(model, J, y, x, u, p.t, p.dt)
jacobian!(model::DiscreteDynamics, J, y, x, u, t, dt) =
    error("User-defined discrete dynamics Jacobian not defined.")

dynamics_error(model::DiscreteDynamics, z2::AbstractKnotPoint, z1::AbstractKnotPoint) =
    discrete_dynamics(model, z1) - state(z2)
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
) = error("User-defined dynamics error Jacobian not defined.")

function propagate_dynamics!(
    ::InPlace,
    model::DiscreteDynamics,
    z2::AbstractKnotPoint,
    z1::AbstractKnotPoint,
)
    discrete_dynamics!(model, getstate(z2), z1)
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