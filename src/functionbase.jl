"""
    AbstractFunction

A function of the form:

```math
y = f(x,u; t)
```
where ``x`` is a state vector of size `n` and ``u`` is a control input of size `m` for some 
dynamical system.  The output ``y`` is of size `p`. 
Here ``t`` represents any number of optional extra parameters to the function (such as time, 
time step, etc.).

The dimensions `n`, `m`, and `p` can 
be queried individual by calling `state_dim`, `control_dim`, and `ouput_dim` or 
collectively via [`dims`](@ref).

# Evaluation 
An `AbstractFunction` can be evaluated in-place by calling any of 

    evaluate!(fun, y, z::AbstractKnotPoint)
    evaluate!(fun, y, x, u, p)
    evaluate!(fun, y, x, u)

The top method should be preferred by generic APIs.

The function can also be evaluated out-of-place (generally returning a 
`StaticArrays.SVector`) by calling any of

    evaluate(fun, z::AbstractKnotPoint)
    evaluate(fun, x, u, p)
    evaluate(fun, x, u)

Alternatively, the user can dispatch on [`FunctionSignature`](@ref) by calling

    evaluate!(::FunctionSignature, fun, y, z::AbstractKnotPoint)

# Jacobians
The Jacobian of ``f(x,u)`` with respect to both ``x`` and ``u`` can be computed 
by calling the following function:

    jacobian!(::FunctionSignature, ::DiffMethod, fun, J, y, z::AbstractKnotPoint)

Where `J` is the Jacobian of size `(p, n + m)` and `y` is provided as an extra storage 
vector for evaluating the Jacobian. Some users may whish to store the function output
value in this vector if evaluating the Jacobian and output value together offers some 
computational savings, but this behavior is left to the user to implement on their 
particular function.

A user-defined Jacobian, called by passing the `UserDefined` [`DiffMethod`](@ref), 
can be implemented for the function by defining any of the following methods for 
the `AbstractFunction`:

    jacobian!(fun, J, y, z::AbstractKnotPoint)
    jacobian!(fun, J, y, x, u, p)
    jacobian!(fun, J, y, x, u)

# Functions of just the state or control
Alternatively, the function can also be limited to an input of just the state or control 
by defining the [`FunctionInputs`](@ref) trait. See trait documentation for more 
information.

# Convenience functions
The methods `fill(fun, v)`, `randn(fun)`, `rand(fun)`, and `zeros(fun)` are defined
that provide a tuple with a state and control vector initialized using the corresponding
function. The data type can be specified as the first argument for `randn`, `rand`, and 
`zeros`.
"""
abstract type AbstractFunction end

"""
    FunctionSignature

Specifies which function signature to call when evaluating the function, must be 
either `StaticReturn` or `InPlace`. The default signature for a function can be 
queried via `default_signature(fun)`.
"""
abstract type FunctionSignature end
struct InPlace <: FunctionSignature end
struct StaticReturn <: FunctionSignature end

"""
    DiffMethod

Represents the method used to evaluate the Jacobian of the function. Allows the 
user to implement multiple methods and switch between them as needed. The following
methods are provided:

* `ForwardAD`: forward-mode automatic differentiation using 
   [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl)
* `FiniteDifference`: finite difference approximation using 
   [FiniteDiff.jl](https://github.com/JuliaDiff/FiniteDiff.jl)
* `UserDefined`: analytical function defined by the user.

The `UserDefined` method can be provided by defining any of the following methods 
on your [`AbstractFunction`](@ref):

    jacobian!(fun, J, y, z::AbstractKnotPoint)
    jacobian!(fun, J, y, x, u, p)
    jacobian!(fun, J, y, x, u)

The other two modes must be explicitly defined on your function, since some functions 
may wish to tailor the method to their particular function.  Efficient default 
implementations can be defined automatically via the [`@autodiff`](@ref) macro.
"""
abstract type DiffMethod end
struct ForwardAD <: DiffMethod end
struct FiniteDifference <: DiffMethod end
struct UserDefined <: DiffMethod end
struct ImplicitFunctionTheorem{D<:DiffMethod} <: DiffMethod
    function ImplicitFunctionTheorem(diff::D) where D <: DiffMethod
        if D <: ImplicitFunctionTheorem
            throw(ArgumentError("DiffMethod for ImplicitFunctionTheorem cannot also be ImplicitFunctionTheorem."))
        end
        new{D}()
    end
end
default_diffmethod(::AbstractFunction) = UserDefined()
default_signature(::AbstractFunction) = StaticReturn()

"""
    dims(fun::AbstractFunction)

Get the tuple `(n,m,p)` where `n` is the dimension of the state vector, `m` is the 
dimension of the control vector, and `p` is the output dimension of the function.
"""
dims(fun::AbstractFunction) = (state_dim(fun), control_dim(fun), output_dim(fun))
state_dim(fun::AbstractFunction) = throw(NotImplementedError("state_dim needs to be implemented for $(typeof(fun))."))
control_dim(fun::AbstractFunction) = throw(NotImplementedError("control_dim needs to be implemented for $(typeof(fun))."))
output_dim(fun::AbstractFunction) = throw(NotImplementedError("output_dim needs to be implemented for $(typeof(fun))"))
errstate_dim(fun::AbstractKnotPoint) = state_dim(fun)
jacobian_width(fun::AbstractFunction) = errstate_dim(fun) + control_dim(fun)

"""
    FunctionInputs

A trait of an [`AbstractFunction`](@ref) that specified the inputs to the function. 
By default, the input is assumed to be both the state and the control vector, along with 
any extra parameters. This trait allows the user to change this assumption, defining 
functions on just the state or control vectors. The number of columns in the Jacobian
will change accordingly, and can be queried via [`input_dim`](@ref). Functions of 
just the state or control are not allowed to take in any extra parameters as arguments.

This trait is queried via [`functioninputs(fun)`](@ref).

The following three options are provided:
* `StateControl`: a function of the form ``y = f(x,u)`` (default)
* `StateOnly`: a function of the form ``y = f(x)``
* `ControlOnly`: a function of the form ``y = f(u)``

When defining methods for these functions, you need to disambiguate methods of the form
    
    evaluate(fun::AbstractFunction, z::AbstractKnotPoint)

from 

    evaluate(fun::MyStateOnlyFunction, x)

It's not enough to annotate `x` as an `AbstractVector` because 
`AbstractKnotPoint <: AbstractVector`. For most cases, the `RobotDynamics.DataVector` 
type should be sufficient to accomplish this. For `StateOnly` or `ControlOnly` functions, 
your methods should look like:

    evaluate(fun::MyFunction, x::RobotDynamics.DataVector)
    evaluate!(fun::MyFunction, y, x::RobotDynamics.DataVector)
    jacobian!(fun::MyFunction, J, y, x::RobotDynamics.DataVector)

"""
abstract type FunctionInputs end
struct StateOnly <: FunctionInputs end
struct ControlOnly <: FunctionInputs end
struct StateControl <: FunctionInputs end

"""
    functioninputs(fun::AbstractFunction)

Get the [`FunctionInputs`](@ref) trait on the function, specifying whether the 
function takes as input both state and control vectors (default), or just the 
state or control vector independently.
"""
functioninputs(fun::AbstractFunction) = StateControl()

"""
    input_dim(fun::AbstractFunction)

Get the dimension of the inputs to the function. Will be equal to `n + m` for 
a `StateVector` function, `n` for a `StateOnly` function, and `m` for a `ControlOnly`
function.
"""
input_dim(fun::AbstractFunction) = input_dim(functioninputs(fun), fun)
input_dim(::StateOnly, fun::AbstractFunction) = state_dim(fun)
input_dim(::ControlOnly, fun::AbstractFunction) = control_dim(fun)
input_dim(::StateControl, fun::AbstractFunction) = state_dim(fun) + control_dim(fun)

# Gets the input to the function given a KnotPoint as a single vector
getinput(::StateControl, z::AbstractKnotPoint) = getdata(z)
getinput(::StateOnly, z::AbstractKnotPoint) = state(z)
getinput(::ControlOnly, z::AbstractKnotPoint) = control(z)

# Gets the input to the function given a KnotPoint as separate arguments
getargs(::StateControl, z::AbstractKnotPoint) = state(z), control(z), getparams(z) 
getargs(::StateOnly, z::AbstractKnotPoint) = (state(z),)
getargs(::ControlOnly, z::AbstractKnotPoint) = (control(z),)

# @inline getinputs(z::AbstractVector) = z
# @inline setinputs!(dest::AbstractVector{<:Real}, src::AbstractVector{<:Real}) = src 
datatype(::Type{<:AbstractFunction}) = Float64

# Top-level command that can be overridden
# Should only be overridden if using hand-written Jacobian methods
evaluate!(fun::AbstractFunction, y, z::AbstractKnotPoint) = evaluate!(functioninputs(fun), fun, y, z) 
evaluate(fun::AbstractFunction, z::AbstractKnotPoint) = evaluate(functioninputs(fun), fun, z) 

evaluate!(inputtype::FunctionInputs, fun::AbstractFunction, y, z::AbstractKnotPoint) = evaluate!(fun, y, getargs(inputtype, z)...)
evaluate(inputtype::FunctionInputs, fun::AbstractFunction, z::AbstractKnotPoint) = evaluate(fun, getargs(inputtype, z)...)

# Strip the parameter
evaluate!(fun::AbstractFunction, y, x, u, p) = evaluate!(fun, y, x, u) 
evaluate(fun::AbstractFunction, x, u, p) = evaluate(fun, x, u) 

# Dispatch on function signature 
evaluate!(::StaticReturn, fun::AbstractFunction, y, args...) = y .= evaluate(fun, args...)
evaluate!(::InPlace, fun::AbstractFunction, y, args...) = evaluate!(fun, y, args...)

# Minimal call that must be implemented
evaluate!(fun::AbstractFunction, y, x, u) = 
    throw(NotImplementedError("User-defined in-place function not implemented for $(typeof(fun))")) 
evaluate(fun::AbstractFunction, x, u) = 
    throw(NotImplementedError("User-defined static return function not implemented for $(typeof(fun))")) 

# Dispatch on FunctionSignature
evaluate!(::StaticReturn, fun::AbstractFunction, y, z::AbstractKnotPoint) = 
    y .= evaluate(fun, z)
evaluate!(::InPlace, fun::AbstractFunction, y, z::AbstractKnotPoint) = 
    evaluate!(fun, y, z)

# Jacobian
# Most generic method: this should the version called by external APIs
jacobian!(::FunctionSignature, ::UserDefined, fun::AbstractFunction, J, y, z) = jacobian!(fun, J, y, z)

# Highest-level function to implement for User-defined Jacobians
jacobian!(fun::AbstractFunction, J, y, z) = jacobian!(functioninputs(fun), fun, J, y, z)

# Dispatches on the input signature to call one of the following:
#  jacobian!(fun, J, y, x, u, p)  # StateControl
#  jacobian!(fun, J, y, x)        # StateOnly
#  jacobian!(fun, J, y, u)        # ControlOnly
jacobian!(inputtype::FunctionInputs, fun::AbstractFunction, J, y, z::AbstractKnotPoint) = jacobian!(fun, J, y, getargs(inputtype, z)...)

# Strip the parameter
jacobian!(fun::AbstractFunction, J, y, x, u, p) = jacobian!(fun, J, y, x, u)

# Minimal call to be implemented for StateControl inputs
jacobian!(fun::AbstractFunction, J, y, x, u) = throw(NotImplementedError("User-defined Jacobian not implemented for $(typeof(fun))")) 

# Jacobian of Jacobian-vector product 
∇jacobian!(::FunctionSignature, ::UserDefined, fun::AbstractFunction, H, b, y, z) = 
    ∇jacobian!(fun, H, b, y, z)
∇jacobian!(fun::AbstractFunction, H, b, y, z::AbstractKnotPoint) = 
    ∇jacobian!(fun, H, b, y, state(z), control(z), getparams(z))
∇jacobian!(fun::AbstractFunction, H, b, y, x, u, p) = 
    ∇jacobian!(fun, H, b, y, state(z), control(z))
∇jacobian!(fun::AbstractFunction, H, b, y, x, u) = 
    throw(UserDefined("User-defined Jacobian of Jacobian-vector product is undefined for $(typeof(fun))"))


# Some convenience methods
Base.rand(::Type{T}, model::AbstractFunction) where {T} = ((@SVector rand(T, state_dim(model))), (@SVector rand(T, control_dim(model))))
Base.randn(::Type{T}, model::AbstractFunction) where {T} = ((@SVector randn(T, state_dim(model))), (@SVector randn(T, control_dim(model))))
Base.zeros(::Type{T}, model::AbstractFunction) where {T} = ((@SVector zeros(T, state_dim(model))), (@SVector zeros(T, control_dim(model))))
Base.fill(model::AbstractFunction, v) = ((@SVector fill(v, state_dim(model))), (@SVector fill(v, control_dim(model))))

Base.rand(model::AbstractFunction) = Base.rand(Float64, model)
Base.randn(model::AbstractFunction) = Base.randn(Float64, model)
Base.zeros(model::AbstractFunction) = Base.zeros(Float64, model)