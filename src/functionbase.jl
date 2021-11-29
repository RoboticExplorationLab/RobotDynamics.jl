abstract type AbstractFunction end

abstract type FunctionSignature end
struct InPlace <: FunctionSignature end
struct StaticReturn <: FunctionSignature end
function_signature(fun::AbstractFunction) = InPlace()

abstract type DiffMethod end
struct ForwardAD <: DiffMethod end
struct FiniteDifference <: DiffMethod end
struct UserDefined <: DiffMethod end
default_diffmethod(::AbstractFunction) = UserDefined()
default_signature(::AbstractFunction) = StaticReturn()

abstract type StateVectorType end
struct EuclideanState <: StateVectorType end
struct RotationState <: StateVectorType end
statevectortype(::Type{<:AbstractFunction}) = EuclideanState()
statevectortype(fun::AbstractFunction) = statevectortype(typeof(fun))

dims(fun::AbstractFunction) = (state_dim(fun), control_dim(fun), output_dim(fun))
state_dim(fun::AbstractFunction) = throw(NotImplementedError("state_dim needs to be implemented for $(typeof(fun))."))
control_dim(fun::AbstractFunction) = throw(NotImplementedError("control_dim needs to be implemented for $(typeof(fun))."))
output_dim(fun::AbstractFunction) = throw(NotImplementedError("output_dim needs to be implemented for $(typeof(fun))"))
errstate_dim(fun::AbstractKnotPoint) = state_dim(fun)

Base.size(fun::AbstractFunction) = (state_dim(fun), control_dim(fun), output_dim(fun))
@inline getinputs(z::AbstractVector) = z
@inline setinputs!(dest::AbstractVector{<:Real}, src::AbstractVector{<:Real}) = src 
inputtype(::Type{<:AbstractFunction}) = Float64

# Top-level command that can be overridden
# Should only be overridden if using hand-written Jacobian methods
evaluate!(fun::AbstractFunction, y, z::AbstractKnotPoint) = evaluate!(fun, y, state(z), control(z), getparams(z))
evaluate(fun::AbstractFunction, z::AbstractKnotPoint) = evaluate(fun, state(z), control(z), getparams(z))

# Strip the parameter
evaluate!(fun::AbstractFunction, y, x, u, p) = evaluate!(fun, y, x, u) 
evaluate(fun::AbstractFunction, x, u, p) = evaluate(fun, x, u) 

# Minimal call that must be implemented
evaluate!(fun::AbstractFunction, y, x, u) = 
    throw(NotImplementedError("User-defined in-place function not implemented for $(typeof(fun))")) 
evaluate(fun::AbstractFunction, x, u) = 
    throw(NotImplementedError("User-defined static return function not implemented for $(typeof(fun))")) 

# Jacobian
jacobian!(::FunctionSignature, ::UserDefined, fun::AbstractFunction, J, y, z) = jacobian!(fun, J, y, z)
jacobian!(fun::AbstractFunction, J, y, z::AbstractKnotPoint) = jacobian!(fun, J, y, state(z), control(z), getparams(z))
jacobian!(fun::AbstractFunction, J, y, x, u, p) = jacobian!(fun, J, y, x, u)
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

# Dispatch on `statevectortype` trait
state_diff(fun::AbstractFunction, x, x0) = state_diff(statevectortype(fun), fun, x, x0)
errstate_dim(fun::AbstractFunction) = errstate_dim(statevectortype(fun), fun)
state_diff_jacobian!(fun::AbstractFunction, J, x) = 
    state_diff_jacobian!(statevectortype(fun), fun, J, x)
∇²differential!(fun::AbstractFunction, ∇G, x, dx) = 
    ∇²differential!(statevectortype(fun), fun, ∇G, x, dx)

# Euclidean state vectors
state_diff(::EuclideanState, fun::AbstractFunction, x, x0) = x - x0
errstate_dim(::EuclideanState, fun::AbstractFunction) = state_dim(fun)
state_diff_jacobian!(::EuclideanState, fun::AbstractFunction, J, x) = J .= I(state_dim(fun))
∇²differential!(::EuclideanState, fun::AbstractFunction, ∇G, x, dx) = ∇G .= 0

# Some convenience methods
Base.randn(::Type{T}, model::AbstractFunction) where {T} = ((@SVector randn(T, state_dim(model))), (@SVector randn(T, control_dim(model))))
Base.zeros(::Type{T}, model::AbstractFunction) where {T} = ((@SVector zeros(T, state_dim(model))), (@SVector zeros(T, control_dim(model))))
Base.rand(::Type{T}, model::AbstractFunction) where {T} = ((@SVector rand(T, state_dim(model))), (@SVector rand(T, control_dim(model))))
Base.fill(model::AbstractFunction, v) = ((@SVector fill(v, state_dim(model))), (@SVector fill(v, control_dim(model))))

Base.randn(model::AbstractFunction) = Base.randn(Float64, model)
Base.zeros(model::AbstractFunction) = Base.zeros(Float64, model)
Base.rand(model::AbstractFunction) = Base.rand(Float64, model)