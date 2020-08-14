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

An abstract subtype of AbstractLinearModel for discrete linear systems that contains LTI and LTV systems. The subtypes of this model automatically implement the discrete_dynamics and 
discrete_jacobian! functions.
"""
abstract type DiscreteLinearModel <: AbstractLinearModel end

"""
    DiscreteLTI<: DiscreteLinearModel

An abstract subtype of DiscreteLinearModel for discrete LTI systems of the following form:
    ``x_{k+1} = Ax_k + Bu_k``
    or 
    ``x_{k+1} = Ax_k + Bu_k + d``

# Interface
All instances of DiscreteLTI should support the following functions:
    get_A(model::DiscreteLTI) 
    get_B(model::DiscreteLTI)

By default, it is assumed that the system is truly linear (eg. not affine). In order to specify systems of the second form:
    is_affine(model::DiscreteLTI) = Val(true)
    get_d(model::DiscreteLTI)

Subtypes of this model should use the integration type DiscreteLinearQuadrature. 

"""
abstract type DiscreteLTI <: DiscreteLinearModel end

"""
    DiscreteLTV<: DiscreteLinearModel

An abstract subtype of DiscreteLinearModel for discrete LTV systems of the following form:
    ``x_{k+1} = A_k x_k + B_k u_k``
    or 
    ``x_{k+1} = A_k x_k + B_k u_k + d_k``

# Interface
All instances of DiscreteLTI should support the following functions:
    get_A(model::DiscreteLTI, k::Integer) 
    get_B(model::DiscreteLTI, k::Integer)

By default, it is assumed that the system is truly linear (eg. not affine). In order to specify systems of the second form:
    is_affine(model::DiscreteLTI) = Val(true)
    get_d(model::DiscreteLTI)

Subtypes of this model should use the integration type DiscreteLinearQuadrature. 

"""
abstract type DiscreteLTV <: DiscreteLinearModel end

abstract type ContinuousLinearModel <: AbstractLinearModel end
abstract type ContinuousLTI <: ContinuousLinearModel end
abstract type ContinuousLTV <: ContinuousLinearModel end

is_affine(::AbstractLinearModel) = Val(false)

is_time_varying(::AbstractLinearModel) = false
is_time_varying(::DiscreteLTV) = true
is_time_varying(::ContinuousLTV) = true

# default to not passing in k
for method ∈ (:get_A, :get_B, :get_d)
    @eval $method(model::AbstractLinearModel, k::Integer) = $method(model)
    @eval $method(model::M) where M <: AbstractLinearModel = throw(ErrorException("$method not implemented for $M")) 
end

abstract type Exponential <: Explicit end
abstract type DiscreteLinearQuadrature <: Explicit end

get_k(t, model::AbstractLinearModel) = is_time_varying(model) ? searchsortedlast(get_times(model), t) : 1
get_times(model::AbstractLinearModel) = throw(ErrorException("get_times not implemented"))

function dynamics(model::ContinuousLinearModel, x, u, t)
    _dynamics(is_affine(model), model, x, u, t)
end

function _dynamics(::Val{true}, model::ContinuousLinearModel, x, u, t)
    k = get_k(t, model)
    return get_A(model, k)*x + get_B(model, k)*u + get_d(model, k)
end

function _dynamics(::Val{false}, model::ContinuousLinearModel, x, u, t)
    k = get_k(t, model)
    return get_A(model, k)*x + get_B(model, k)*u
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

function discrete_dynamics(::Type{DiscreteLinearQuadrature}, model::DiscreteLinearModel, x::StaticVector, u::StaticVector, t, dt)
    _discrete_dynamics(is_affine(model), model, x, u, t, dt)
end

function _discrete_dynamics(::Val{true}, model::DiscreteLinearModel, x::StaticVector, u::StaticVector, t, dt)
    k = get_k(t, model)
    get_A(model, k)*x + get_B(model, k)*u + get_d(model, k)
end

function _discrete_dynamics(::Val{false}, model::DiscreteLinearModel, x::StaticVector, u::StaticVector, t, dt)
    k = get_k(t, model)
    get_A(model, k)*x + get_B(model, k)*u
end

function discrete_jacobian!(::Type{DiscreteLinearQuadrature}, ∇f, model::DiscreteLinearModel, z::AbstractKnotPoint{<:Any,n,m}) where {n,m}
    t = z.t
    k = get_k(t, model)
    
    n = state_dim(model)
    m = control_dim(model)

    ix = 1:n
    iu = n .+ (1:m)
    ∇f[ix,ix] .= get_A(model, k)
    ∇f[ix,iu] .= get_B(model, k)

    nothing
end

discrete_dynamics(::Type{Exponential}, model::M, x, u, t, dt) where M = throw(ErrorException("Exponential integration not defined for model type $M"))
discrete_dynamics(::Type{Exponential}, model::ContinuousLinearModel, x, u, t, dt) = throw(ErrorException("TODO: implement exponential integration"))