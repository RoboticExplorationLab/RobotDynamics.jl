"""
    LinearizedModel{M,L,T} <: AbstractModel

A container for the linearized model that holds the full nonlinear model, the linearized model, and the
trajectory of linearization points. The same dynamics and jacobian functions can still be called on the
`LinearizedModel` type.
"""
struct LinearizedModel{M, L, T} <: AbstractModel
    model::M
    linmodel::L
    trajectory::T
end

"""
    LinearizedModel(model::AbstractModel, linmodel::LinearModel, traj::AbstractTrajectory)

A constructor for the linearized model type. The inputted model should be the full continuous dynamics
model to be linearized.
"""
LinearizedModel(model::M, linmodel::L, traj::T) where {M <: AbstractModel, L <: LinearModel, T <: AbstractTrajectory} = 
    LinearizedModel{M, L, T}(model, linmodel, traj)


dynamics(model::LinearizedModel, x, u, t=0.0) = dynamics(model.linmodel, x, u, t)
discrete_dynamics(::Type{PassThrough}, model::LinearizedModel, x, u, t, dt) = 
    discrete_dynamics(PassThrough, model.linmodel, x, u, t, dt)

jacobian!(∇f::AbstractMatrix, model::LinearizedModel, z::AbstractKnotPoint) = 
    jacobian!(∇f, model.linmodel, z)

discrete_jacobian!(::Type{PassThrough}, ∇f, model::LinearizedModel, z::AbstractKnotPoint) = 
    discrete_jacobian!(PassThrough, ∇f, model.linmodel, z)

"""
    linearize_and_discretize(::Type{Q}, nonlinear_model::AbstractModel, trajectory::AbstractTrajectory) where {Q<:Explicit}

Linearize `nonlinear_model` about `trajectory` and discretize with given integration type `Q`. Returns a [`LinearizedModel`](@ref) 
"""
function linearize_and_discretize(::Type{Q}, nonlinear_model::AbstractModel, trajectory::AbstractTrajectory; is_affine=false) where {Q<:Explicit}
    n,m = state_dim(nonlinear_model), control_dim(nonlinear_model)
    times = length(trajectory) > 1 ? get_times(trajectory) : 1:0

    dt = trajectory[1].dt
    linmodel = LinearModel(n, m; is_affine=is_affine, times=times, dt=dt)
    model = LinearizedModel(nonlinear_model, linmodel, trajectory)
    linearize_and_discretize!(Q, model, nonlinear_model, trajectory)

    model
end

"""
    linearize_and_discretize!(::Type{Q}, model::LinearizedModel, nonlinear_model::AbstractModel, trajectory::AbstractTrajectory) where {Q<:Explicit}

Linearize `nonlinear_model` about `trajectory` and discretize with given integration type `Q`. Put results into `linear_model`. 

!!! warning
    Ensure that the time varying `linear_model` has the same length as the trajectory and that the knot points in the trajectory correspond
    with the times returned by `get_times`.
"""
# should clear up model vs nonlinearmodel vs trajectory
function linearize_and_discretize!(::Type{Q}, model::LinearizedModel, nonlinear_model::AbstractModel, trajectory::AbstractTrajectory) where {Q<:Explicit}
    N = length(trajectory)

    @assert is_discrete(model.linmodel) == true
    @assert is_timevarying(model.linmodel) == true
    @assert get_times(trajectory) == (model.linmodel.times)

    if is_timevarying(model.linmodel)
        for i=1:N-1
            linearize_and_discretize!(Q, model, nonlinear_model, trajectory[i])
        end
    else
        linearize_and_discretize!(Q, model, nonlinear_model, trajectory[1])
    end
end

"""
    linearize_and_discretize!(::Type{Q}, model::LinearizedModel, nonlinear_model::AbstractModel, z::AbstractKnotPoint) where {Q<:Explicit}

Linearize `nonlinear_model` about `z` and discretize with given integration type `Q`. Put results into `linear_model`. 
"""
function linearize_and_discretize!(::Type{Q}, model::LinearizedModel, nonlinear_model::AbstractModel, z::AbstractKnotPoint) where {Q<:Explicit}
    @assert is_discrete(model.linmodel) == true

    _linearize_and_discretize!(Q, model.linmodel, nonlinear_model, z)
end

# TODO: add special implementations here for models with rotations, rigid bodies

function _linearize_and_discretize!(::Type{Q}, linmodel::LinearModel, nonlinear_model::AbstractModel, z::AbstractKnotPoint) where  Q <: Explicit
    ix, iu = z._x, z._u
    t = z.t
    dt = z.dt
    x̄ = z.z[ix]
    ū = z.z[iu]
    
    # dispatch so as not to always using StaticArray implementation?
    F = DynamicsJacobian(nonlinear_model)
    jacobian!(F, nonlinear_model, z)
    A_c = get_A(F)
    B_c = get_B(F)

    k = get_k(linmodel, t)

    if is_affine(linmodel)
        d_c = dynamics(nonlinear_model, z) - A_c*x̄ - B_c*ū
        _discretize!(Q, linmodel, A_c, B_c, d_c, k, dt)
    else
        _discretize!(Q, linmodel, A_c, B_c, linmodel.d, k, dt)
    end

    nothing
end


"""
    discretize!(::Type{Q}, discrete_model::LinearModel, continuous_model::LinearModel; dt=0.0) where {Q<:Explicit}

Discretize `continuous_model` with given integration type. Put results into `discrete_model`. 
"""
function discretize!(::Type{Q}, discrete_model::LinearModel, continuous_model::LinearModel) where {Q<:Explicit}
    @assert is_timevarying(continuous_model) == is_timevarying(discrete_model)
    @assert is_affine(continuous_model) == is_affine(discrete_model)

    @assert is_discrete(discrete_model)
    @assert !is_discrete(continuous_model)

    if is_timevarying(continuous_model)
        @assert all(continuous_model.times .== discrete_model.times)

        times = continuous_model.times
        N = length(times)

        for i=1:N-1
            dt = times[i+1] - times[i]

            d = is_affine(continuous_model) ? continuous_model.d[k] : continuous_model.d
            _discretize!(Q, discrete_model, continuous_model.A[k], continuous_model.B[k], d, i, dt)
        end
    else
        d = is_affine(continuous_model) ? continuous_model.d[k] : continuous_model.d
        _discretize!(Q, discrete_model, continuous_model.A[1], continuous_model.B[1], d, 1, discrete_model.dt)
    end
end

function _discretize!(::Type{Exponential}, discrete_model::LinearModel, A::AbstractMatrix, B::AbstractMatrix, d::AbstractVector, k::Integer, dt)
    n = size(A, 1)
    I = oneunit(SizedMatrix{n, n})
    m = size(B, 2)

    if is_affine(discrete_model)
        continuous_system = zero(SizedMatrix{(2*n)+m, (2*n)+m})
        continuous_system[SUnitRange(1,n), SUnitRange(1,n)] .= A
        continuous_system[SUnitRange(1,n), SUnitRange(n+1,n+m)] .= B
        continuous_system[SUnitRange(1,n), SUnitRange(n+m+1,2*n+m)] .= I

        discrete_system = exp(continuous_system*dt)
        A_d = discrete_system[SUnitRange(1,n), SUnitRange(1,n)]
        B_d = discrete_system[SUnitRange(1,n), SUnitRange(n+1,n+m)]

        discrete_model.A[k] = A_d
        discrete_model.B[k] = B_d

        D_d = discrete_system[SUnitRange(1,n), SUnitRange(n+m+1,2*n+m)]
        discrete_model.d[k] = D_d*d
    else
        continuous_system = zero(SizedMatrix{n+m, n+m})
        continuous_system[SUnitRange(1,n), SUnitRange(1,n)] .= A
        continuous_system[SUnitRange(1,n), SUnitRange(n+1,n+m)] .= B

        discrete_system = exp(continuous_system*dt)
        A_d = discrete_system[SUnitRange(1,n), SUnitRange(1,n)]
        B_d = discrete_system[SUnitRange(1,n), SUnitRange(n+1,n+m)]

        discrete_model.A[k] = A_d
        discrete_model.B[k] = B_d
    end

    nothing
end

function _discretize!(::Type{RK2}, discrete_model::LinearModel, A::AbstractMatrix, B::AbstractMatrix, d::AbstractVector, k::Integer, dt) 
    A_d = oneunit(typeof(A)) + A*dt + A^2*dt^2/2
    B_d = B*dt + A*B*dt^2/2
    
    discrete_model.A[k] = A_d 
    discrete_model.B[k] = B_d

    if is_affine(discrete_model)
        d_d = d*dt + A*d*dt^2/2
        discrete_model.d[k] = d_d
    end

    nothing
end

function _discretize!(::Type{RK3}, discrete_model::LinearModel, A::AbstractMatrix, B::AbstractMatrix, d::AbstractVector, k::Integer, dt) 
    A_d = oneunit(typeof(A)) + A*dt + A^2*dt^2/2 + A^3*dt^3/6
    B_d = B*dt + A*B*dt^2/2 + A^2*B*dt^3/6
    
    discrete_model.A[k] = A_d 
    discrete_model.B[k] = B_d

    if is_affine(discrete_model)
        d_d = d*dt + A*d*dt^2/2 + A^2*d*dt^3/6
        discrete_model.d[k] = d_d
    end

    nothing
end

function _discretize!(::Type{RK4}, discrete_model::LinearModel, A::AbstractMatrix, B::AbstractMatrix, d::AbstractVector, k::Integer, dt) 
    A_d = oneunit(typeof(A)) + A*dt + A^2*dt^2/2 + A^3*dt^3/6 + A^4*dt^4/24
    B_d = B*dt + A*B*dt^2/2 + A^2*B*dt^3/6 + A^3*B*dt^4/24
    
    discrete_model.A[k] = A_d 
    discrete_model.B[k] = B_d

    if is_affine(discrete_model)
        d_d = d*dt + A*d*dt^2/2 + A^2*d*dt^3/6 + A^3*d*dt^4/24
        discrete_model.d[k] = d_d
    end

    nothing
end

function _discretize!(::Type{Q}, discrete_model::LinearModel, A::AbstractMatrix, B::AbstractMatrix, d::AbstractVector, k::Integer, dt) where Q <: Explicit
    throw(ErrorException("Discretization special case not yet defined for $Q."))
end