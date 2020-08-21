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
    @eval ($method)(model::AbstractLinearModel, mat::AbstractArray, k::Integer) = ($method)(model, mat)
    @eval ($method)(model::M, mat::AbstractArray) where M <: AbstractLinearModel = throw(ErrorException("$($method) not implemented for $M")) 
end

macro create_discrete_ltv(name, n, m, N, is_affine=false)
    if is_affine   
        _ltv_affine(name, n, m, N, :DiscreteLTV)
    else
        _ltv_non_affine(name, n, m, N, :DiscreteLTV)
    end
end

macro create_continuous_ltv(name, n, m, N, is_affine=false)
    if is_affine   
        _ltv_affine(name, n, m, N, :ContinuousLTV)
    else
        _ltv_non_affine(name, n, m, N, :ContinuousLTV)
    end
end

function _ltv_non_affine(name, n, m, N, supertype)
    struct_exp = quote
        struct ($name){T} <: ($supertype)
            A::Vector{SMatrix{$n,$n,T,($n)^2}}
            B::Vector{SMatrix{$n,$m,T,($n*$m)}}
            times::Vector{T}
        end
    end
    function_def = quote
        RobotDynamics.is_affine(::($name)) = Val(false)
        RobotDynamics.control_dim(::($name)) = $m
        RobotDynamics.state_dim(::($name)) = $n
        RobotDynamics.get_A(model::($name), k::Integer) = model.A[k]
        RobotDynamics.get_B(model::($name), k::Integer) = model.B[k]
        RobotDynamics.get_times(model::($name)) = model.times
        RobotDynamics.set_A!(model::($name), A::AbstractArray, k::Integer) = model.A[k] = A
        RobotDynamics.set_B!(model::($name), B::AbstractArray, k::Integer) = model.B[k] = B
        RobotDynamics.set_times!(model::($name), times::AbstractVector) = model.times .= times

        function ($name)()
            A_vec = [@SMatrix zeros($n, $n) for _ = 1:($N-1)]
            B_vec = [@SMatrix zeros($n, $m) for _ = 1:($N-1)]
            times = zeros($N)

            return ($name){Float64}(A_vec, B_vec, times)
        end
    end

    call_exp = quote
        ($name)()
    end

    esc(Expr(:toplevel,
            struct_exp,
            function_def,
            call_exp))
end

function _ltv_affine(name, n, m, N, supertype)
    struct_exp = quote
        struct ($name){T} <: ($supertype)
            A::Vector{SMatrix{$n,$n,T,($n)^2}}
            B::Vector{SMatrix{$n,$m,T,($n*$m)}}
            d::Vector{SVector{$n,T}}
            times::Vector{T}
        end
    end
    function_def = quote
        RobotDynamics.is_affine(::($name)) = Val(true)
        RobotDynamics.control_dim(::($name)) = $m
        RobotDynamics.state_dim(::($name)) = $n
        RobotDynamics.get_A(model::($name), k::Integer) = model.A[k]
        RobotDynamics.get_B(model::($name), k::Integer) = model.B[k]
        RobotDynamics.get_d(model::($name), k::Integer) = model.d[k]
        RobotDynamics.get_times(model::($name)) = model.times
        RobotDynamics.set_A!(model::($name), A::AbstractArray, k::Integer) = model.A[k] = A
        RobotDynamics.set_B!(model::($name), B::AbstractArray, k::Integer) = model.B[k] = B
        RobotDynamics.set_d!(model::($name), d::AbstractArray, k::Integer) = model.d[k] = d
        RobotDynamics.set_times!(model::($name), times::AbstractVector) = model.times .= times

        function ($name)()
            A_vec = [@SMatrix zeros($n, $n) for _ = 1:($N-1)]
            B_vec = [@SMatrix zeros($n, $m) for _ = 1:($N-1)]
            d_vec = [@SVector zeros($n) for _ = 1:($N-1)]
            times = zeros($N)

            return ($name){Float64}(A_vec, B_vec, d_vec, times)
        end
    end

    call_exp = quote
        ($name)()
    end

    esc(Expr(:toplevel,
            struct_exp,
            function_def,
            call_exp))
end

macro create_discrete_lti(name, n, m, is_affine=false)
    if is_affine   
        _lti_affine(name, n, m, :DiscreteLTI)
    else
        _lti_non_affine(name, n, m, :DiscreteLTI)
    end
end

macro create_continuous_lti(name, n, m, is_affine=false)
    if is_affine   
        _lti_affine(name, n, m, :ContinuousLTI)
    else
        _lti_non_affine(name, n, m, :ContinuousLTI)
    end
end

function _lti_affine(name, n, m, supertype)
    struct_exp = quote
        struct ($name){T} <: ($supertype)
            A::Base.RefValue{SMatrix{$n,$n,T,($n)^2}}
            B::Base.RefValue{SMatrix{$n,$m,T,($n*$m)}}
            d::Base.RefValue{SVector{$n,T}}
        end
    end
    function_def = quote
        RobotDynamics.is_affine(::($name)) = Val(true)
        RobotDynamics.control_dim(::($name)) = $m
        RobotDynamics.state_dim(::($name)) = $n
        RobotDynamics.get_A(model::($name)) = model.A[]
        RobotDynamics.get_B(model::($name)) = model.B[]
        RobotDynamics.get_d(model::($name)) = model.d[]
        RobotDynamics.set_A!(model::($name), A::AbstractArray) = model.A[] = A
        RobotDynamics.set_B!(model::($name), B::AbstractArray) = model.B[] = B
        RobotDynamics.set_d!(model::($name), d::AbstractArray) = model.d[] = d

        function ($name)()
            A = Ref(@SMatrix zeros($n, $n))
            B = Ref(@SMatrix zeros($n, $m))
            d = Ref(@SVector zeros($n))

            return ($name){Float64}(A, B, d)
        end
    end

    call_exp = quote
        ($name)()
    end

    esc(Expr(:toplevel,
            struct_exp,
            function_def,
            call_exp))
end

function _lti_non_affine(name, n, m, supertype)
    struct_exp = quote
        struct ($name){T} <: ($supertype)
            A::Base.RefValue{SMatrix{$n,$n,T,($n)^2}}
            B::Base.RefValue{SMatrix{$n,$m,T,($n*$m)}}
        end
    end
    function_def = quote
        RobotDynamics.is_affine(::($name)) = Val(false)
        RobotDynamics.control_dim(::($name)) = $m
        RobotDynamics.state_dim(::($name)) = $n
        RobotDynamics.get_A(model::($name)) = model.A[]
        RobotDynamics.get_B(model::($name)) = model.B[]
        RobotDynamics.set_A!(model::($name), A::AbstractArray) = model.A[] = A
        RobotDynamics.set_B!(model::($name), B::AbstractArray) = model.B[] = B

        function ($name)()
            A = Ref(@SMatrix zeros($n, $n))
            B = Ref(@SMatrix zeros($n, $m))

            return ($name){Float64}(A, B)
        end
    end

    call_exp = quote
        ($name)()
    end

    esc(Expr(:toplevel,
            struct_exp,
            function_def,
            call_exp))
end