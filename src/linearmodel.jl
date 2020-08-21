#=
Type tree:
                 AbstractLinearModel <: AbstractModel
                ↙                                    ↘
         DiscreteLinearModel                   ContinuousLinearModel
         ↙               ↘                       ↙                ↘
 DiscreteLTI           DiscreteLTV       ContinuousLTI          ContinuousLTV
                                
=#


"""
    AbstractLinearModel <: AbstractModel

A general supertype for implementing a linear model. The subtypes of this model allow the automatic implementation of dynamics and jacobian functions for improved performance.
"""
abstract type AbstractLinearModel <: AbstractModel end

"""
    DiscreteLinearModel <: AbstractLinearModel

An abstract subtype of `AbstractLinearModel` for discrete linear systems that contains LTI and LTV systems. The subtypes of this model automatically implement the `discrete_dynamics` and 
`discrete_jacobian!` functions.
"""
abstract type DiscreteLinearModel <: AbstractLinearModel end

"""
    DiscreteLTI<: DiscreteLinearModel

An abstract subtype of `DiscreteLinearModel` for discrete LTI systems of the following form:
    ``x_{k+1} = Ax_k + Bu_k``
    or 
    ``x_{k+1} = Ax_k + Bu_k + d``  

# Interface
All instances of `DiscreteLTI` should support the following functions:

    get_A(model::DiscreteLTI) 
    get_B(model::DiscreteLTI)

By default, it is assumed that the system is truly linear (eg. not affine). In order to specify affine systems:

    is_affine(model::DiscreteLTI) = Val(true)
    get_d(model::DiscreteLTI)

Subtypes of this model should use the integration type `DiscreteSystemQuadrature`. 

"""
abstract type DiscreteLTI <: DiscreteLinearModel end

"""
    DiscreteLTV<: DiscreteLinearModel

An abstract subtype of `DiscreteLinearModel` for discrete LTV systems of the following form:  
    ``x_{k+1} = A_k x_k + B_k u_k``  
    or 
    ``x_{k+1} = A_k x_k + B_k u_k + d_k``  

# Interface
All instances of `DiscreteLTV` should support the following functions:

    get_A(model::DiscreteLTV, k::Integer) 
    get_B(model::DiscreteLTV, k::Integer)

By default, it is assumed that the system is truly linear (eg. not affine). In order to specify affine systems:

    is_affine(model::DiscreteLTV) = Val(true)
    get_d(model::DiscreteLTV, k::Integer)
    get_times(model::DiscreteLTV)

Subtypes of this model should use the integration type `DiscreteSystemQuadrature`. 

"""
abstract type DiscreteLTV <: DiscreteLinearModel end

"""
    ContinuousLinearModel <: AbstractLinearModel

An abstract subtype of `AbstractLinearModel` for continuous linear systems that contains LTI and LTV systems. The subtypes of this model automatically implement the `dynamics` and 
`jacobian!` functions. For trajectory optimization problems, it will generally be faster to integrate your system matrices externally and implement a `DiscreteLinearModel`. This
reduces unnecessary calls to the dynamics function and increases speed.
"""
abstract type ContinuousLinearModel <: AbstractLinearModel end

"""
    ContinuousLTI<: ContinuousLinearModel

An abstract subtype of `ContinuousLinearModel` for continuous LTI systems of the following form:  
    ``ẋ = Ax + Bu``  
    or
     ``ẋ = Ax + Bu + d``  

# Interface
All instances of `ContinuousLTI` should support the following functions:

    get_A(model::ContinuousLTI) 
    get_B(model::ContinuousLTI)

By default, it is assumed that the system is truly linear (eg. not affine). In order to specify affine systems:

    is_affine(model::ContinuousLTI) = Val(true)
    get_d(model::ContinuousLTI)

"""
abstract type ContinuousLTI <: ContinuousLinearModel end

"""
    ContinuousLTV<: ContinuousLinearModel

An abstract subtype of `ContinuousLinearModel` for continuous LTV systems of the following form:  
    ``ẋ = A_k x + B_k u``  
    or 
    ``ẋ = A_k x + B_k u + d_k``  

# Interface
All instances of `ContinuousLTV` should support the following functions:

    get_A(model::ContinuousLTV, k::Integer) 
    get_B(model::ContinuousLTV, k::Integer)

By default, it is assumed that the system is truly linear (eg. not affine). In order to specify affine systems:

    is_affine(model::ContinuousLTV) = Val(true)
    get_d(model::ContinuousLTV, k::Integer)
    get_times(model::ContinuousLTV)

"""
abstract type ContinuousLTV <: ContinuousLinearModel end

is_affine(::AbstractLinearModel) = Val(false)

is_time_varying(::AbstractLinearModel) = false
is_time_varying(::DiscreteLTV) = true
is_time_varying(::ContinuousLTV) = true

# default to not passing in k
for method ∈ (:get_A, :get_B, :get_d)
    @eval ($method)(model::AbstractLinearModel, k::Integer) = ($method)(model)
    @eval ($method)(model::M) where M <: AbstractLinearModel = throw(ErrorException("$($method) not implemented for $M")) 
end

abstract type DiscreteSystemQuadrature <: Explicit end

get_k(t, model::AbstractLinearModel) = is_time_varying(model) ? searchsortedlast(get_times(model), t) : 1

"""
    get_times(model::AbstractLinearModel)

This function should be overloaded by the user to return the list of times for a time varying linear system. The index
k of the time varying system gotten using `searchsortedlast` and the returned list. This only needs to be defined for
time varying systems.

    get_times(model::MyModel) = [0.0, 0.05, 0.1]

In the above example, the system matrices for k = 1 are defined for t ∈ [0.0, 0.05). The system matrices for
k = 2 are defined for t ∈ [0.05, 0.01). 
"""
get_times(model::AbstractLinearModel) = throw(ErrorException("get_times not implemented"))

function dynamics(model::ContinuousLinearModel, x, u, t)
    _dynamics(is_affine(model), model, x, u, t)
end

function _dynamics(::Val{true}, model::ContinuousLinearModel, x, u, t)
    k = get_k(t, model)
    return get_A(model, k)*x .+ get_B(model, k)*u .+ get_d(model, k)
end

function _dynamics(::Val{false}, model::ContinuousLinearModel, x, u, t)
    k = get_k(t, model)
    A = get_A(model, k)
    B = get_B(model, k)
    return A*x + B*u
end

function jacobian!(∇f::AbstractMatrix, model::ContinuousLinearModel, z::AbstractKnotPoint)
	t = z.t
    k = get_k(t, model)

    n = state_dim(model)
    m = control_dim(model)

    ∇f[1:n, 1:n] .= get_A(model, k)
    ∇f[1:n, (n+1):(n+m)] .= get_B(model, k)
    true
end

function discrete_dynamics(::Type{DiscreteSystemQuadrature}, model::DiscreteLinearModel, x::StaticVector, u::StaticVector, t, dt)
    _discrete_dynamics(is_affine(model), model, x, u, t, dt)
end

function _discrete_dynamics(::Val{true}, model::DiscreteLinearModel, x::StaticVector, u::StaticVector, t, dt)
    k = get_k(t, model)
    get_A(model, k)*x .+ get_B(model, k)*u .+ get_d(model, k)
end

function _discrete_dynamics(::Val{false}, model::DiscreteLinearModel, x::StaticVector, u::StaticVector, t, dt)
    k = get_k(t, model)
    get_A(model, k)*x + get_B(model, k)*u
end

function discrete_jacobian!(::Type{DiscreteSystemQuadrature}, ∇f, model::DiscreteLinearModel, z::AbstractKnotPoint{<:Any,n,m}) where {n,m}
    t = z.t
    k = get_k(t, model)
    ix = 1:n
    iu = n .+ (1:m)
    ∇f[ix,ix] .= get_A(model, k)
    ∇f[ix,iu] .= get_B(model, k)

    nothing
end

# TODO: create macro to automatically create Linear Model

abstract type Exponential <: Explicit end
abstract type Euler <: Explicit end
const RK1 = Euler

# default to not passing in k here
for method ∈ (:set_A!, :set_B!, :set_d!)
    @eval ($method)(model::DiscreteLinearModel, mat::AbstractArray, k::Integer) = ($method)(model, mat)
    @eval ($method)(model::M, mat::AbstractArray) where M <: DiscreteLinearModel = throw(ErrorException("$($method) not implemented for $M")) 
end

linearize_and_discretize!(linear_model::DiscreteLinearModel, nonlinear_model::AbstractModel, trajectory::AbstractTrajectory) = 
    linearize_and_discretize!(DEFAULT_Q, linear_model, nonlinear_model, trajectory)

function linearize_and_discretize!(::Type{Q}, linear_model::DiscreteLinearModel, nonlinear_model::AbstractModel, trajectory::AbstractTrajectory) where {Q<:Explicit}
    for knot_point in trajectory
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

    k = get_k(linear_model, t)

    set_A!(linear_model, A, k)
    set_B!(linear_model, B, k)
    set_d!(linear_model, d, k)
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

    k = get_k(linear_model, t)

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

    k = get_k(linear_model, t)

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

# macro create_discrete_ltv(name, n, m, N; is_affine=false)
#     struct_exp = quote
#         struct $name <: DiscreteLTV
#             A::SMatrix
#             B::SMatrix
#         end
#     end
#     function_def = quote
#         RobotDynamics.is_affine(::($name)) = Val(is_affine)
#         RobotDynamics.control_dim(::($name)) = $m
#         RobotDynamics.state_dim(::($name)) = $n
#         RobotDynamics.get_A(model::($name), k::Integer) = model.A[k]
#         RobotDynamics.get_B(model::($name), k::Integer) = model.B[k]
#         RobotDynamics.get_d(model::($name), k::Integer) = model.d[k]
#         RobotDynamics.set_A!(model::($name), A::AbstractMatrix, k::Integer) = model.A[k] = A
#         RobotDynamics.set_B!(model::($name), B::AbstractMatrix, k::Integer) = model.B[k] = B
#     end

#     if is_affine
#         println("should have put in an affine")
#     end
#     esc(Expr(:toplevel,
#             struct_exp,
#             function_def))
# end

# macro create_continuous_ltv(name, is_affine, n, m, N)

# end

# macro create_discrete_lti(name, is_affine, n, m)

# end

# macro create_continuous_lti(name, is_affine, n, m)

# end

# function create_linear_model_tv(name, is_affine, n, m, N, supertype)

# end

# function create_linear_model_ti(name, is_affine, n, m, supertype)

# end