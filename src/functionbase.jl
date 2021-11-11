abstract type AbstractFunction end

abstract type FunctionSignature end
struct InPlace <: FunctionSignature end
struct StaticReturn <: FunctionSignature end
function_signature(fun::AbstractFunction) = InPlace()

abstract type DiffMethod end
struct ForwardAD <: DiffMethod end
struct FiniteDifference <: DiffMethod end
struct UserDefined <: DiffMethod end
diff_method(fun::AbstractFunction) = UserDefined

state_dim(fun::AbstractFunction) = state_dim(typeof(fun))
control_dim(fun::AbstractFunction) = control_dim(typeof(fun))
output_dim(fun::AbstractFunction) = error("output_dim not implemented!")

Base.size(fun::AbstractFunction) = (state_dim(fun), control_dim(fun), output_dim(fun))
@inline getinputs(z::AbstractVector) = z
@inline setinputs!(dest::AbstractVector{<:Real}, src::AbstractVector{<:Real}) = src 

# Top-level command that can be overridden
# Should only be overridden if using hand-written Jacobian methods
evaluate!(fun::AbstractFunction, y, z::KnotPoint) = evaluate!(fun, y, state(z), control(z), getparams(z))
evaluate(fun::AbstractFunction, z::KnotPoint) = evaluate(fun, state(z), control(z), getparams(z))

# Strip the parameter
evaluate!(fun::AbstractFunction, y, x, u, p) = evaluate!(fun, y, x, u) 
evaluate(fun::AbstractFunction, x, u, p) = evaluate(fun, x, u) 

# Minimal call that must be implemented
evaluate!(::AbstractFunction, y, x, u) = error("In-place function evaluation not defined yet!")
evaluate(::AbstractFunction, x, u) = error("Static return function evaluation not defined yet!")

inputtype(::Type{<:AbstractFunction}) = Float64

jacobian!(::FunctionSignature, ::UserDefined, fun::AbstractFunction, J, y, z) = jacobian!(fun, J, y, z)
jacobian!(fun::AbstractFunction, J, y, z::KnotPoint) = jacobian!(fun, J, y, state(z), control(z), getparams(z))
jacobian!(fun::AbstractFunction, J, y, x, u, p) = jacobian!(fun, J, y, x, u)
jacobian!(fun, J, y, x, u) = error("User-defined Jacobian hasn't been specified.")
