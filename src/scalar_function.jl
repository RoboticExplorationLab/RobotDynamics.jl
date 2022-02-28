"""
    ScalarFunction <: AbstractFunction

Represents a scalar function of the form:

```math
c = f(x,u)
```
where ``c \\in \\mathbb{R}``.

# Evaluation
Since the function return a scalar, both `evaluate` and `evaluate!` call the 
same function methods. To avoid confusion, `evaluate` should always be preferred 
when working with a `ScalarFunction`. To use, simply implement one of the following 
methods:

    evaluate(fun, x, u, p)
    evaluate(fun, x, u)

where `p` is tuple of parameters.

# Differentiation
First and second-order derivatives of scalar functions are commonly referred to as 
gradients and Hessians. We use the convention that a gradient is a 1-dimensional array 
(i.e. an `AbstractVector` with size `(n,)`) while the Jacobian of a scalar function is a 
row vector (i.e. an `AbstractMatrix` with size `(1,n)`). Theses methods can be 
called using:

    gradient!(::DiffMethod, fun, grad, z)
    hessian!(::DiffMethod, fun, hess, z)

Which allows the user to dispatch on the [`DiffMethod`](@ref). These methods can 
also be called by calling the more generic `jacobian` and `∇jacobian!` methods:

    jacobian!(sig, diff, fun, J, y, z)
    ∇jacobian!(sig, diff, fun, H, b, y, z)

where the length of `y`, and `b` is 1, and `b[1] == one(eltype(b))`.

To implement `UserDefined` methods, implement any one of the following gradient methods:

    gradient!(::UserDefined, fun, grad, z)
    gradient!(fun, grad, z)
    gradient!(fun, grad, x, u, p)
    gradient!(fun, grad, x, u)

and any one of the following Hessian methods:

    hessian!(::UserDefined, fun, hess, z)
    hessian!(fun, hess, z)
    hessian!(fun, hess, x, u, p)
    hessian!(fun, hess, x, u)

"""
abstract type ScalarFunction <: AbstractFunction end
output_dim(fun::ScalarFunction) = 1

# Inplace reverts to scalar return
evaluate!(fun::ScalarFunction, y, x, u, p) = evaluate(fun, x, u, p)

# Gradients
function jacobian!(sig::FunctionSignature, diff::DiffMethod, fun::ScalarFunction, J, y, z) 
    @assert length(y) == 1
    gradient!(sig, diff, fun, J', z)
end

gradient!(::FunctionSignature, diff::DiffMethod, fun::ScalarFunction, grad, z) = 
    gradient!(diff, fun, grad, z)
gradient!(diff::UserDefined, fun::ScalarFunction, grad, z) = 
    gradient!(fun, grad, z)
gradient!(fun::AbstractFunction, grad, z::AbstractKnotPoint) = 
    gradient!(fun, grad, state(z), control(z), getparams(z))
gradient!(fun::AbstractFunction, grad, x, u, p) = 
    gradient!(fun, grad, x, u)
gradient!(fun::AbstractFunction, grad, x, u) = 
    throw(NotImplementedError("Gradient not implemented for scalar function $(typeof(fun))"))

# Hessian
function ∇jacobian!(sig::FunctionSignature, diff::DiffMethod, fun::ScalarFunction, H, b, y, z)
    @assert b[1] ≈ one(eltype(b)) 
    @assert length(y) == 1
    hessian!(sig, diff, fun, H, z)
end

hessian!(::FunctionSignature, diff::DiffMethod, fun::ScalarFunction, hess, z) = 
    hessian!(diff, fun, hess, z)
hessian!(diff::UserDefined, fun::ScalarFunction, hess, z) = hessian!(fun, hess, z)
hessian!(fun::ScalarFunction, hess, z::AbstractKnotPoint) = 
    hessian!(fun, hess, state(z), control(z), getparams(z))
hessian!(fun::ScalarFunction, hess, x, u, p) = 
    hessian!(fun, hess, x, u)
hessian!(fun::ScalarFunction, hess, x, u) = 
    throw(NotImplementedError("Hessian not implemented for scalar function $(typeof(fun))"))
