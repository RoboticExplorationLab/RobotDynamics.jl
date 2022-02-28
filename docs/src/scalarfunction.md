# Scalar Functions
```@meta
CurrentModule = RobotDynamics
```
Instead of working with vector-valued functions like dynamics functions, we often need 
to define scalar functions that accept our state and control vectors, such as cost / 
objective / reward functions. This page provides the API for working with these 
types of functions, represented by the abstract [`ScalarFunction`](@ref) type, which 
is a specialization of an [`AbstractFunction`](@ref) with an output dimension of 1.

```@docs
ScalarFunction
```