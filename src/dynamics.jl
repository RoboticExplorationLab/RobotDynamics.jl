## Dynamics
abstract type AbstractModel <: AbstractFunction end
output_dim(model::AbstractModel) = state_dim(model)

abstract type ContinuousDynamics <: AbstractModel end
@inline dynamics(model::ContinuousDynamics, z::AbstractKnotPoint) =
    dynamics(model, state(z), control(z), time(z))
@inline dynamics(model::ContinuousDynamics, xdot, z::AbstractKnotPoint) =
    dynamics!(model, xdot, state(z), control(z), time(z))

@inline evaluate(model::ContinuousDynamics, x, u, p) =
    dynamics(model, x, u, p.t) 
@inline evaluate!(model::ContinuousDynamics, ẋ, x, u, p) =
    dynamics!(model, ẋ, x, u, p.t) 

@inline dynamics(model::ContinuousDynamics, x, u, t) = dynamics(model, x, u)
@inline dynamics!(model::ContinuousDynamics, ẋ, x, u, t) = dynamics!(model, ẋ, x, u)
@inline jacobian!(::FunctionSignature, ::UserDefined, model::ContinuousDynamics, J, ẋ, z) =
    jacobian!(model, J, ẋ, state(z), control(z), time(z))