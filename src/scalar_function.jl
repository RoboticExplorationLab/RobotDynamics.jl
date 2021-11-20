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
gradient!(diff::DiffMethod, fun::ScalarFunction, grad, z) = 
    throw(NotImplementedError("Gradient not implemented for scalar function $(typeof(fun))"))

# Hessian
function ∇jacobian!(sig::FunctionSignature, diff::DiffMethod, fun::ScalarFunction, H, b, y, z)
    @assert b[1] ≈ 1.0
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
