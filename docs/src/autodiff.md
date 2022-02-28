# Differentiation API
```@meta
CurrentModule = RobotDynamics
```
RobotDynamics allows users to define different methods for obtaining derivatives of their 
functions. This is accomplished via the [`DiffMethod`](@ref) trait, which by default 
has three options:
* `ForwardAD`: forward-mode automatic differentiation using 
   [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl)
* `FiniteDifference`: finite difference approximation using 
   [FiniteDiff.jl](https://github.com/JuliaDiff/FiniteDiff.jl)
* `UserDefined`: analytical function defined by the user.

Almost every Jacobian is queried using the same form:

    jacobian!(sig, diff, fun, J, y, z)

where `sig` is a [`FunctionSignature`](@ref), `diff` is a [`DiffMethod`](@ref), `fun` is 
an [`AbstractFunction`](@ref), `J` is the Jacobian, `y` is a vector the size of the 
output of the function, and `z` is an [`AbstractKnotPoint`]. Users are free to add more 
[`DiffMethod`](@ref) types to dispatch on their own differentiation method. 

By default, no differentiation methods are provided for an [`AbstractFunction`](@ref), 
allowing the user to choose what methods they want defined, and to allow customization 
of the particular method to their function. However, we do provide the following macro 
to provide efficient implementations for `ForwardAD` and `FiniteDifference`:

```@docs
@autodiff
```
