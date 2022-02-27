
"""
    StateVectorType

A trait defined on an `AbstractFunction`, describing how state vectors are composed 
and how the error state, or the "difference" between two vectors, is computed.
These "error states" should be Euclidean vectors, and can have a different dimension than 
the original state vectors. 

The following types are provided by RobotDynamics:

* [`EuclideanState`](@ref)
* [`RotationState`](@ref)

## Defining a new state vector type
To define a custom state vector type (e.g. `NewStateType`), you need to implement 
the function that calculates the difference between state vectors, as well the 
"error state Jacobian" and it's derivative.
First, you must specify the dimension of the error state by defining

    errstate_dim(::NewStateType, fun::AbstractFunction)

Then you define method that calculates the error state between two state vectors: 

    state_diff!(::NewStateType, model, dx, x, x0)

where `dx` should have dimension `e = output_dim(model)` and `x` and `x0` should 
have dimension `n = state_dim(model)`.

The other piece of information we need is the "error state Jacobian." This is the 
first-order approximation of the mapping from local error state values to the true 
state vector. Let `x ⊗ x0` be the composition of two state vectors, and let `φ` be 
the function that maps a Euclidean error state `δx` to a state vector. The error 
state Jacobian is the Jacobian of `x ⊗ φ(δx)` with respect to `δx`, taking the limit 
as `δx → 0`. The resulting function is solely a function of the state `x`, and is 
implemented with the following function:

    errstate_jacobian!(::NewStateType, model, G, x)

where `G` has dimensions `(n,e)`.

We'll also need the derivative of this function. For computational efficiency, 
we compute the Jacobian of the error state Jacobian transposed with some vector `x̄`,
where `x̄` is state vector of size `(n,)`. More precisely, it is the Hessian of 

```math
\\bar{x}^T (x \\otimes \\varphi(\\delta x))
````
This function is calculated using

    ∇errstatef_jacobian!(::NewStateType, model, ∇G, x, xbar)

where `∇G` is of size `(e,e)`.
"""
abstract type StateVectorType end

"""
    EuclideanState

The space of standard Euclidean vectors, where composition is element-wise addition, 
and the error state is just element-wise subtraction.
"""
struct EuclideanState <: StateVectorType end

"""
    RotationState

The space of vectors composes of Euclidean states intermixed with 3D rotations, 
represented using any 3 or 4-parameter representation (usually unit quaternions).
The error state Jacobians and error state calculation are all calculating using 
[Rotations.jl](https://github.com/JuliaGeometry/Rotations.jl).
"""
struct RotationState <: StateVectorType end

"""
    statevectortype

Queries the StateVectorType trait on a function.
"""
statevectortype(::Type{<:AbstractFunction}) = EuclideanState()
statevectortype(fun::AbstractFunction) = statevectortype(typeof(fun))

# Dispatch on `statevectortype` trait
"""
    errstate_dim

The length of the Euclidean error state for the given state vector type.
"""
errstate_dim(fun::AbstractFunction) = errstate_dim(statevectortype(fun), fun)

"""
    state_diff!(fun, dx, x, x0)

Calculate the error state between `x` and `x0`. The Euclidean error state `dx` should 
have dimension equal to `output_dim(fun)` while `x` and `x0` should have dimension 
equal to `state_dim(fun)`. For Euclidean states, this is just 

```math
\\delta x = x - x_0
```
"""
state_diff!(fun::AbstractFunction, dx, x, x0) = state_diff!(statevectortype(fun), fun, dx, x, x0)

state_diff(fun::AbstractFunction, x, x0) = state_diff(statevectortype(fun), fun, x, x0)

"""
    errstate_jacobian!

The Jacobian of 

```math
x \\otimes \\varphi(\\delta x)
```
with respect to ``\\delta x``.
Here ``\\otimes`` is the composition function for two elements of the state space,
and ``\\varphi(\\delta x)`` is the function that elements of the Euclidean error state to 
the state vector space.
"""
errstate_jacobian!(fun::AbstractFunction, J, z::AbstractKnotPoint) = 
    errstate_jacobian!(statevectortype(fun), fun, J, state(z))
errstate_jacobian!(fun::AbstractFunction, J, x) = 
    errstate_jacobian!(statevectortype(fun), fun, J, x)

"""
    ∇errstate_jacobian!

The Hessian of

```math
\\bar{x}^T (x \\otimes \\varphi(\\delta x))
```
with respect to ``\\delta x``.
Here ``\\otimes`` is the composition function for two elements of the state space,
and ``\\varphi(\\delta x)`` is the function that elements of the Euclidean error state to 
the state vector space. The vector ``\\bar{x}`` is some element of the state space.
"""
∇errstate_jacobian!(fun::AbstractFunction, J, z::AbstractKnotPoint, dx) =   
    ∇errstate_jacobian!(statevectortype(fun), fun, J, state(z), dx)

∇errstate_jacobian!(fun::AbstractFunction, J, x, dx) =   
    ∇errstate_jacobian!(statevectortype(fun), fun, J, x, dx)

# Euclidean state vectors
errstate_dim(::EuclideanState, fun::AbstractFunction) = state_dim(fun)
state_diff!(::EuclideanState, fun::AbstractFunction, dx, x, x0) = dx .= x .- x0
state_diff(::EuclideanState, fun::AbstractFunction, x, x0) = x - x0
errstate_jacobian!(::EuclideanState, fun::AbstractFunction, J, x) = J .= I(state_dim(fun))
∇errstate_jacobian!(::EuclideanState, fun::AbstractFunction, ∇G, x, dx) = ∇G .= 0