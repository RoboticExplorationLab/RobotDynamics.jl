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
jacobian_width(fun::AbstractFunction) = errstate_dim(fun) + control_dim(fun)

abstract type FunctionInputs end
struct StateOnly <: FunctionInputs end
struct ControlOnly <: FunctionInputs end
struct StateControl <: FunctionInputs end
functioninputs(fun::AbstractFunction) = StateControl()
input_dim(fun::AbstractFunction) = input_dim(functioninputs(fun), fun)
input_dim(::StateOnly, fun::AbstractFunction) = state_dim(fun)
input_dim(::ControlOnly, fun::AbstractFunction) = control_dim(fun)
input_dim(::StateControl, fun::AbstractFunction) = state_dim(fun) + control_dim(fun)

getinput(::StateControl, z::AbstractKnotPoint) = getdata(z)
getinput(::StateOnly, z::AbstractKnotPoint) = state(z)
getinput(::ControlOnly, z::AbstractKnotPoint) = control(z)

getargs(::StateControl, z::AbstractKnotPoint) = state(z), control(z), getparams(z) 
getargs(::StateOnly, z::AbstractKnotPoint) = (state(z),)
getargs(::ControlOnly, z::AbstractKnotPoint) = (control(z),)

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

# Dispatch on `statevectortype` trait
state_diff!(fun::AbstractFunction, dx, x, x0) = state_diff!(statevectortype(fun), fun, dx, x, x0)
state_diff(fun::AbstractFunction, x, x0) = state_diff(statevectortype(fun), fun, x, x0)
errstate_dim(fun::AbstractFunction) = errstate_dim(statevectortype(fun), fun)
state_diff_jacobian!(fun::AbstractFunction, J, z::AbstractKnotPoint) = 
    state_diff_jacobian!(statevectortype(fun), fun, J, state(z))
state_diff_jacobian!(fun::AbstractFunction, J, x) = 
    state_diff_jacobian!(statevectortype(fun), fun, J, x)
∇²differential!(fun::AbstractFunction, ∇G, x, dx) = 
    ∇²differential!(statevectortype(fun), fun, ∇G, x, dx)

# Euclidean state vectors
state_diff!(::EuclideanState, fun::AbstractFunction, dx, x, x0) = dx .= x .- x0
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