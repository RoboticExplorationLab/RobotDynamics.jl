"""
    LinearizedModel{M,L,T} <: AbstractModel

A container for the linearized model that holds the full nonlinear model, the linearized model, and the
trajectory of linearization points. The same dynamics and jacobian functions can still be called on the
`LinearizedModel` type.

# Constructors
    LinearizedModel(nonlinear_model::AbstractModel, Z::AbstractTrajectory; kwargs...)
    LinearizedModel(nonlinear_model::AbstractModel, [z::AbstractKnotPoint]; kwargs...)

Linearizes `nonlinear_model` about the trajectory `Z` or a single point `z`. If not specified,
`z` is defined as the state and control defined by `zeros(nonlinear_model)`.

Linearization is, by default, on the continuous system.  

# Keyword Arguments
* `is_affine` - Linearize the system with an affine term, such that the new state is the same as the original state. See below for more details.
* `dt` - Time step. If not provided, defaults to the value in `Z` or `z`. Must be specified and non-zero for the system to be discretized. If `dt = NaN`, then the dt will be inferred from the trajectory (useful for variable step sizes).
* `integration` - An explicit integration method. Must also specify a non-zero dt. 

If `is_affine = false`, the dynamics are defined as:
``f(x,u) \\approx f(x_0, u_0) + A \\delta x + B \\delta u``

which defines the error state ``\\delta x = x - x_0``
    as the state of the linearized system. Here ``A`` and ``B`` are the partial derivative
    of the dynamics with respect to the state and control, respectively.

If `is_affine = true`, the form is an affine function of the form
``f(x,u) \\approx A x + B u + d``

where ``d = f(x_0,u_0) - A x_0 - B u_0``, which maintains the same definition of the state.
"""
struct LinearizedModel{M, Q, L, T, D} <: AbstractLinearModel
    model::M
    linmodel::L
    Z::T
    F::D
    function LinearizedModel(::Type{Q}, model::AbstractModel, linmodel::LinearModel, 
            Z::AbstractTrajectory) where Q <: QuadratureRule
        if !is_discrete(linmodel)
            @assert Q == Continuous
        end
        n,m = size(linmodel)
        F = DynamicsJacobian(model)
        new{typeof(model), Q, typeof(linmodel), typeof(Z), typeof(F)}(
            model, linmodel, Z, F
        )
    end
end

for method in (:state_dim, :control_dim, :is_discrete, :is_affine, :is_timevarying)
    @eval $(method)(model::LinearizedModel) = $(method)(model.linmodel)
end
integration(::LinearizedModel{<:Any,Q}) where Q = Q

get_A(model::LinearizedModel, k=1) = get_A(model.linmodel, k)
get_B(model::LinearizedModel, k=1) = get_B(model.linmodel, k)
get_k(model::LinearizedModel, t) = get_k(model.linmodel, t)

"""
"""
function LinearizedModel(nonlinear_model::AbstractModel, Z::AbstractTrajectory; 
        integration::Type{Q}=DEFAULT_Q, 
        dt=0.0,
        is_affine=false
    ) where {Q<:Explicit}
    n,m = state_dim(nonlinear_model), control_dim(nonlinear_model)

    is_discrete = dt != zero(dt)
    if is_discrete
        if !isnan(dt)
            set_dt!(Z, dt)
        end
        Q0 = Q
    else
        Q0 = Continuous
        if Q != DEFAULT_Q
            @warn "Discarding input for integration, since the system in continuous. Must specify a non-zero dt if you want discretize the system."
        end
    end

    if !isnan(dt) && is_discrete
        set_dt!(Z, dt)
    elseif !isnan(dt) && !is_discrete
    end

    times = length(Z) > 1 ? get_times(Z) : 1:0

    linmodel = LinearModel(n, m; is_affine=is_affine, times=times, dt=dt)
    
    model = LinearizedModel(Q0, nonlinear_model, linmodel, Z)

    linearize!(Q0, model)

    model
end

# single point linearization
function LinearizedModel(nonlinear_model::AbstractModel, 
        z::KnotPoint=KnotPoint(zeros(nonlinear_model)...,0.,0.); 
        kwargs...
    ) where {Q<:Explicit}
    LinearizedModel(nonlinear_model, Traj([z]); kwargs...)
end

# Pass dynamics function through to linear model
@inline dynamics(model::LinearizedModel, x, u, t=0.0) = dynamics(model.linmodel, x, u, t)
@inline discrete_dynamics(::Type{PassThrough}, model::LinearizedModel, x, u, t, dt) = 
    discrete_dynamics(PassThrough, model.linmodel, x, u, t, dt)

@inline jacobian!(∇f::AbstractMatrix, model::LinearizedModel, z::AbstractKnotPoint) = 
    jacobian!(∇f, model.linmodel, z)

@inline discrete_jacobian!(::Type{PassThrough}, ∇f, model::LinearizedModel, z::AbstractKnotPoint) =
    discrete_jacobian!(PassThrough, ∇f, model.linmodel, z)

"""
    update_trajectory!(model::LinearizedModel, Z::AbstractTrajectory, integration::=DEFAULT_Q)

Updates the trajectory inside of the `model` and relinearizes (and discretizes for discrete 
    models) the model about the new trajectory.
"""
function update_trajectory!(model::LinearizedModel{<:Any,Q}, Z::AbstractTrajectory) where {Q<:QuadratureRule}
    model.Z .= Z
    linearize!(Q, model)
end


# linearize a single knot point
"""
    linearize!(model::LinearizedModel, integration=DEFAULT_Q)

Linearize the nonlinear model `model.model` about `model.Z`, storing the result in
`model.linmodel`.

Linearization defaults to the form:
``f(x,u) \\approx f(x_0, u_0) + A \\delta x + B \\delta u``

if `is_affine(model.linmodel == false`, which defines the error state ``\\delta x = x - x_0``
    as the state of the linearized system. Here ``A`` and ``B`` are the partial derivative
    of the dynamics with respect to the state and control, respectively.

Otherwise, the form is an affine function of the form
``f(x,u) \\approx A x + B u + d``

where ``d = f(x_0,u_0) - A x_0 - B u_0``, which maintains the same definition of the state.

"""
function linearize!(::Type{Q}, model::LinearizedModel) where Q <: QuadratureRule 
    nlmodel = model.model
    linmodel = model.linmodel

    for k in eachindex(model.Z)
        if is_discrete(linmodel)
            discretize!(Q, model, k) 
        else
            @assert Q == Continuous
            z = model.Z[k]
            F = model.F 
            jacobian!(F, nlmodel, z)
            if is_affine(model)
                linmodel.d[k] .= dynamics(model, z) - F * z.z
            end
            linmodel.A[k] .= get_A(F)
            linmodel.B[k] .= get_B(F)
        end
    end
end


"""
    discretize!(::Type{Q}, model::LinearizedModel, k)

Discretize the linearized model at time step k, using integration `Q`. 
"""
function discretize!(::Type{Q}, model::LinearizedModel, k::Int) where Q <: Explicit
    z = model.Z[k]
    discrete_jacobian!(Q, model.F, model.model, z)
    if is_affine(model)
        d = model.linmodel.d[k]
        d .= discrete_dynamics(Q, model.model, z) - model.F * z.z
    end

    # Copy to LinearModel
    model.linmodel.A[k] .= model.F.A
    model.linmodel.B[k] .= model.F.B
end

function discretize!(::Type{Exponential}, model::LinearizedModel, k::Int)
    n,m = size(model.linmodel)
    z = model.Z[k]
    E = model.linmodel.E
    F = model.F

    # Calculate continuous Jacobian of nonlinear system
    jacobian!(F, model.model, z)

    # Calculate Matrix exponential
    matrix_exponential!(E, get_A(F), get_B(F), z.dt)

    # Copy discrete system
    ix,iu = 1:n, n .+ (1:m)
    A = model.linmodel.A[k]
    B = model.linmodel.B[k]
    A .= E[ix,ix]
    B .= E[ix,iu]


    if is_affine(model)
        D = E[ix, (n+m) .+ (1:n)]
        d0 = dynamics(model.model, z)  # nonlinear dynamics
        d0 = d0 - F * z.z              # continuous affine term

        d = model.linmodel.d[k]
        mul!(d, D, d0)
    end
end

