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

abstract type Exponential <: Explicit end

function _discretize(::Type{Exponential}, ::Val{false}, model::ContinuousLinearModel, k::Integer, dt)
    A_c = get_A(model, k)
    B_c = get_B(model, k)
    n = size(A_c, 1)
    m = size(B_c, 2)

    continuous_system = zero(SizedMatrix{n+m, n+m})
    continuous_system[1:n, 1:n] .= A_c
    continuous_system[1:n, n .+ (1:m)] .= B_c

    discrete_system = exp(continuous_system*dt)
    A_d = discrete_system[StaticArrays.SUnitRange(1,n), StaticArrays.SUnitRange(1,n)]
    B_d = discrete_system[StaticArrays.SUnitRange(1,n), StaticArrays.SUnitRange(n+1,n+m)]

    return (A_d, B_d)
end

function _discretize(::Type{Exponential}, ::Val{true}, model::ContinuousLinearModel, k::Integer, dt)
    A_c = get_A(model, k)
    B_c = get_B(model, k)
    n = size(A_c, 1)
    D_c = oneunit(SizedMatrix{n, n})
    m = size(B_c, 2)

    continuous_system = zero(SizedMatrix{(2*n)+m, (2*n)+m})
    continuous_system[1:n, 1:n] .= A_c
    continuous_system[1:n, n .+ (1:m)] .= B_c
    continuous_system[1:n, n + m .+ (1:n)] .= D_c

    discrete_system = exp(continuous_system*dt)
    A_d = discrete_system[StaticArrays.SUnitRange(1,n), StaticArrays.SUnitRange(1,n)]
    B_d = discrete_system[StaticArrays.SUnitRange(1,n), StaticArrays.SUnitRange(n+1,n+m)]
    D_d = discrete_system[StaticArrays.SUnitRange(1,n), StaticArrays.SUnitRange(n+m+1,2*n+m)]

    return (A_d, B_d, D_d)
end