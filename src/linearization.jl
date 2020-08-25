"""
    linearize_and_discretize!(linear_model::DiscreteLTV, nonlinear_model::AbstractModel, trajectory::AbstractTrajectory)

Linearize `nonlinear_model` about `trajectory` and discretize with given default integration type (`RK3`). Put results into `linear_model`. 

!!! warning
    Ensure that the time varying `linear_model` has the same length as the trajectory and that the knot points in the trajectory correspond
    with the times returned by `get_times`.
"""
linearize_and_discretize!(linear_model::DiscreteLTV, nonlinear_model::AbstractModel, trajectory::AbstractTrajectory) = 
    linearize_and_discretize!(DEFAULT_Q, linear_model, nonlinear_model, trajectory)

"""
    linearize_and_discretize!(::Type{Q}, linear_model::DiscreteLTV, nonlinear_model::AbstractModel, trajectory::AbstractTrajectory) where {Q<:Explicit}

Linearize `nonlinear_model` about `trajectory` and discretize with given integration type. Put results into `linear_model`. 

!!! warning
    Ensure that the time varying `linear_model` has the same length as the trajectory and that the knot points in the trajectory correspond
    with the times returned by `get_times`.
"""
function linearize_and_discretize!(::Type{Q}, linear_model::DiscreteLTV, nonlinear_model::AbstractModel, trajectory::AbstractTrajectory) where {Q<:Explicit}
    N = length(trajectory)

    @assert get_times(trajectory) == (get_times(linear_model))

    for i=1:N-1
        linearize_and_discretize!(Q, linear_model, nonlinear_model, trajectory[i])
    end
end

"""
    linearize_and_discretize!(::Type{Q}, linear_model::DiscreteLinearModel, nonlinear_model::AbstractModel, z::AbstractKnotPoint) where {Q<:Explicit}

Linearize `nonlinear_model` about `z` and discretize with given integration type. Put results into `linear_model`. 
"""
function linearize_and_discretize!(::Type{Q}, linear_model::DiscreteLinearModel, nonlinear_model::AbstractModel, z::AbstractKnotPoint) where {Q<:Explicit}
    _linearize_and_discretize!(Q, is_affine(linear_model), linear_model, nonlinear_model, z)
end

# TODO: add special implementations here for models with rotations, rigid bodies

function _linearize_and_discretize!(::Type{Q}, ::Val{T}, linear_model::DiscreteLinearModel, nonlinear_model::AbstractModel, z::AbstractKnotPoint) where {Q<:Explicit, T}
    ix, iu = z._x, z._u
    t = z.t
    x̄ = z.z[ix]
    ū = z.z[iu]
    
    F = DynamicsJacobian(nonlinear_model)
    discrete_jacobian!(Q, F, nonlinear_model, z)
    A = get_A(F)
    B = get_B(F)

    k = get_k(t, linear_model)

    set_A!(linear_model, A, k)
    set_B!(linear_model, B, k)

    if T
        d = discrete_dynamics(Q, nonlinear_model, z) - A*x̄ - B*ū
        set_d!(linear_model, d, k)
    end
end

function _linearize_and_discretize!(::Type{Exponential}, ::Val{T}, linear_model::DiscreteLinearModel, nonlinear_model::AbstractModel, z::AbstractKnotPoint) where T
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

    k = get_k(t, linear_model)

    if T
        d_c = dynamics(nonlinear_model, z) - A_c*x̄ - B_c*ū
        _discretize!(Exponential, Val(true), linear_model, A_c, B_c, d_c, k, dt)
    else
        _discretize!(Exponential, Val(false), linear_model, A_c, B_c, k, dt)
    end

    nothing
end


"""
    discretize!(::Type{Q}, discrete_model::DiscreteLinearModel, continuous_model::ContinuousLinearModel; dt=0.0) where {Q<:Explicit}

Discretize `continuous_model` with given integration type. Put results into `discrete_model`. 

Infers discretization timestep using `get_times` for time varying models. 

!!! warning
    Time invariant models must specify keyword argument for discretization timestep `dt`.
"""
function discretize!(::Type{Q}, discrete_model::DiscreteLinearModel, continuous_model::ContinuousLinearModel; dt=0.0) where {Q<:Explicit}
    @assert is_time_varying(continuous_model) == is_time_varying(discrete_model)
    @assert is_affine(continuous_model) == is_affine(discrete_model)

    if is_time_varying(continuous_model)
        @assert all(get_times(continuous_model) .== get_times(discrete_model))

        N = length(get_times(continuous_model))
        times = get_times(continuous_model)

        for i=1:N-1
            dt = times[i+1] - times[i]

            _discretize!(Q, is_affine(continuous_model), discrete_model, continuous_model, i, dt)
        end
    else
        _discretize!(Q, is_affine(continuous_model), discrete_model, continuous_model, 1, dt)
    end
end

_discretize!(::Type{Q}, ::Val{false}, discrete_model::DiscreteLinearModel, continuous_model::ContinuousLinearModel, k::Integer, dt) where {Q<:Explicit} = 
    _discretize!(Q, Val(false), discrete_model, get_A(continuous_model, k), get_B(continuous_model, k), k, dt)

_discretize!(::Type{Q}, ::Val{true}, discrete_model::DiscreteLinearModel, continuous_model::ContinuousLinearModel, k::Integer, dt) where {Q<:Explicit} = 
    _discretize!(Q, Val(true), discrete_model, get_A(continuous_model, k), get_B(continuous_model, k), get_d(continuous_model, k), k, dt)

function _discretize!(::Type{Exponential}, ::Val{true}, discrete_model::DiscreteLinearModel, A::AbstractMatrix, B::AbstractMatrix, d::AbstractVector, k::Integer, dt)
    n = size(A, 1)
    I = oneunit(SizedMatrix{n, n})
    m = size(B, 2)

    continuous_system = zero(SizedMatrix{(2*n)+m, (2*n)+m})
    continuous_system[SUnitRange(1,n), SUnitRange(1,n)] .= A
    continuous_system[SUnitRange(1,n), SUnitRange(n+1,n+m)] .= B
    continuous_system[SUnitRange(1,n), SUnitRange(n+m+1,2*n+m)] .= I

    discrete_system = exp(continuous_system*dt)
    A_d = discrete_system[SUnitRange(1,n), SUnitRange(1,n)]
    B_d = discrete_system[SUnitRange(1,n), SUnitRange(n+1,n+m)]

    set_A!(discrete_model, A_d, k)
    set_B!(discrete_model, B_d, k)

    D_d = discrete_system[SUnitRange(1,n), SUnitRange(n+m+1,2*n+m)]
    set_d!(discrete_model, D_d*d, k)

    nothing
end

function _discretize!(::Type{Exponential}, ::Val{false}, discrete_model::DiscreteLinearModel, A::AbstractMatrix, B::AbstractMatrix, k::Integer, dt)
    n = size(A, 1)
    m = size(B, 2)

    continuous_system = zero(SizedMatrix{n+m, n+m})
    continuous_system[SUnitRange(1,n), SUnitRange(1,n)] .= A
    continuous_system[SUnitRange(1,n), SUnitRange(n+1,n+m)] .= B

    discrete_system = exp(continuous_system*dt)
    A_d = discrete_system[SUnitRange(1,n), SUnitRange(1,n)]
    B_d = discrete_system[SUnitRange(1,n), SUnitRange(n+1,n+m)]

    set_A!(discrete_model, A_d, k)
    set_B!(discrete_model, B_d, k)

    nothing
end


function _discretize!(::Type{Euler}, ::Val{T}, discrete_model::DiscreteLinearModel, A::AbstractMatrix, B::AbstractMatrix, d::AbstractVector, k::Integer, dt) where T
    A_d = oneunit(typeof(A)) + A*dt
    B_d = B*dt
    
    set_A!(discrete_model, A_d, k)
    set_B!(discrete_model, B_d, k)

    if T
        d_d = d*dt
        set_d!(discrete_model, d_d, k)
    end

    nothing
end

function _discretize!(::Type{RK2}, ::Val{T}, discrete_model::DiscreteLinearModel, A::AbstractMatrix, B::AbstractMatrix, d::AbstractVector, k::Integer, dt) where T
    A_d = oneunit(typeof(A)) + A*dt + A^2*dt^2/2
    B_d = B*dt + A*B*dt^2/2
    
    set_A!(discrete_model, A_d, k)
    set_B!(discrete_model, B_d, k)

    if T
        d_d = d*dt + A*d*dt^2/2
        set_d!(discrete_model, d_d, k)
    end

    nothing
end
