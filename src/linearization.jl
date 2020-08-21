linearize_and_discretize!(linear_model::DiscreteLinearModel, nonlinear_model::AbstractModel, trajectory::AbstractTrajectory) = 
    linearize_and_discretize!(DEFAULT_Q, linear_model, nonlinear_model, trajectory)

function linearize_and_discretize!(::Type{Q}, linear_model::DiscreteLinearModel, nonlinear_model::AbstractModel, trajectory::AbstractTrajectory) where {Q<:Explicit}
    N = length(trajectory)
    for i=1:N-1
        knot_point = trajectory[i]
        linearize_and_discretize!(Q, linear_model, nonlinear_model, knot_point)
    end
end

function linearize_and_discretize!(::Type{Q}, linear_model::DiscreteLinearModel, nonlinear_model::AbstractModel, z::AbstractKnotPoint) where {Q<:Explicit}
    _linearize_and_discretize!(Q, is_affine(linear_model), linear_model, nonlinear_model, z)
end

# TODO: add special implementations here for models with rotations, rigid bodies

function _linearize_and_discretize!(::Type{Q}, ::Val{true}, linear_model::DiscreteLinearModel, nonlinear_model::AbstractModel, z::AbstractKnotPoint) where {Q<:Explicit}
    ix, iu = z._x, z._u
    t = z.t
    x̄ = z.z[ix]
    ū = z.z[iu]
    
    F = DynamicsJacobian(nonlinear_model)
    discrete_jacobian!(Q, F, nonlinear_model, z)
    A = get_A(F)
    B = get_B(F)
    d = discrete_dynamics(Q, nonlinear_model, z) - A*x̄ - B*ū

    k = get_k(t, linear_model)

    set_A!(linear_model, A, k)
    set_B!(linear_model, B, k)
    set_d!(linear_model, d, k)
end


function _linearize_and_discretize!(::Type{Q}, ::Val{false}, linear_model::DiscreteLinearModel, nonlinear_model::AbstractModel, z::AbstractKnotPoint) where {Q<:Explicit}
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
end


function _linearize_and_discretize!(::Type{Exponential}, ::Val{true}, linear_model::DiscreteLinearModel, nonlinear_model::AbstractModel, z::AbstractKnotPoint)
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
    d_c = dynamics(nonlinear_model, z) - A_c*x̄ - B_c*ū

    k = get_k(t, linear_model)

    _discretize!(Exponential, Val(true), linear_model, A_c, B_c, d_c, k, dt)

    nothing
end

function _linearize_and_discretize!(::Type{Exponential}, ::Val{false}, linear_model::DiscreteLinearModel, nonlinear_model::AbstractModel, z::AbstractKnotPoint)
    ix, iu = z._x, z._u
    t = z.t
    dt = z.dt
    
    # dispatch so as not to always using StaticArray implementation?
    F = DynamicsJacobian(nonlinear_model)
    jacobian!(F, nonlinear_model, z)
    A_c = get_A(F)
    B_c = get_B(F)

    k = get_k(t, linear_model)

    _discretize!(Exponential, Val(false), linear_model, A_c, B_c, k, dt)

    nothing
end

function discretize!(::Type{Q}, discrete_model::DiscreteLinearModel, continuous_model::ContinuousLinearModel; dt=0.05) where {Q<:Explicit}
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

function _discretize!(::Type{Exponential}, ::Val{false}, discrete_model::DiscreteLinearModel, A::AbstractMatrix, B::AbstractMatrix, k::Integer, dt)
    n = size(A, 1)
    m = size(B, 2)

    continuous_system = zero(SizedMatrix{n+m, n+m})
    continuous_system[1:n, 1:n] .= A
    continuous_system[1:n, n .+ (1:m)] .= B

    discrete_system = exp(continuous_system*dt)
    A_d = discrete_system[StaticArrays.SUnitRange(1,n), StaticArrays.SUnitRange(1,n)]
    B_d = discrete_system[StaticArrays.SUnitRange(1,n), StaticArrays.SUnitRange(n+1,n+m)]

    set_A!(discrete_model, A_d, k)
    set_B!(discrete_model, B_d, k)

    nothing
end

function _discretize!(::Type{Exponential}, ::Val{true}, discrete_model::DiscreteLinearModel, A::AbstractMatrix, B::AbstractMatrix, d::AbstractVector, k::Integer, dt)
    n = size(A, 1)
    I = oneunit(SizedMatrix{n, n})
    m = size(B, 2)

    continuous_system = zero(SizedMatrix{(2*n)+m, (2*n)+m})
    continuous_system[1:n, 1:n] .= A
    continuous_system[1:n, n .+ (1:m)] .= B
    continuous_system[1:n, n + m .+ (1:n)] .= I

    discrete_system = exp(continuous_system*dt)
    A_d = discrete_system[StaticArrays.SUnitRange(1,n), StaticArrays.SUnitRange(1,n)]
    B_d = discrete_system[StaticArrays.SUnitRange(1,n), StaticArrays.SUnitRange(n+1,n+m)]
    D_d = discrete_system[StaticArrays.SUnitRange(1,n), StaticArrays.SUnitRange(n+m+1,2*n+m)]

    set_A!(discrete_model, A_d, k)
    set_B!(discrete_model, B_d, k)
    set_d!(discrete_model, D_d*d, k)

    nothing
end

function _discretize!(::Type{Euler}, ::Val{false}, discrete_model::DiscreteLinearModel, A::AbstractMatrix, B::AbstractMatrix, k::Integer, dt)
    A_d = oneunit(typeof(A)) + A*dt
    B_d = B*dt
    
    set_A!(discrete_model, A_d, k)
    set_B!(discrete_model, B_d, k)
end

function _discretize!(::Type{Euler}, ::Val{true}, discrete_model::DiscreteLinearModel, A::AbstractMatrix, B::AbstractMatrix, d::AbstractVector, k::Integer, dt)
    A_d = oneunit(typeof(A)) + A*dt
    B_d = B*dt
    d_d = d*dt
    
    set_A!(discrete_model, A_d, k)
    set_B!(discrete_model, B_d, k)
    set_d!(discrete_model, d_d, k)
end

# TODO: check this
function _discretize!(::Type{RK2}, ::Val{false}, discrete_model::DiscreteLinearModel, A::AbstractMatrix, B::AbstractMatrix, k::Integer, dt)
    A_d = oneunit(typeof(A)) + A*dt + A^2*dt^2/2
    B_d = B*dt + A*B*dt^2/2
    
    set_A!(discrete_model, A_d, k)
    set_B!(discrete_model, B_d, k)
end

function _discretize!(::Type{RK2}, ::Val{true}, discrete_model::DiscreteLinearModel, A::AbstractMatrix, B::AbstractMatrix, d::AbstractVector, k::Integer, dt)
    A_d = oneunit(typeof(A)) + A*dt + A^2*dt^2/2
    B_d = B*dt + A*B*dt^2/2
    d_d = d*dt + A*d*dt^2/2
    
    set_A!(discrete_model, A_d, k)
    set_B!(discrete_model, B_d, k)
    set_d!(discrete_model, d_d, k)
end
