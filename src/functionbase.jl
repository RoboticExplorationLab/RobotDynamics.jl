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

jacobian!(::InPlace, ::UserDefined, fun::AbstractFunction, J, y, z) =
    jacobian!(fun, J, y, z)
jacobian!(::StaticReturn, ::UserDefined, fun::AbstractFunction, J, y, z) =
    jacobian!(fun, J, z)

