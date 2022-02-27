# `AbstractFunction` API
```@meta
CurrentModule = RobotDynamics
```

RobotDynamics sets up a unified framework for evaluating functions accepting a 
state and control vector as inputs, not just dynamics functions. This can be used 
by downstream packages to define functions such as cost functions or constraints.
This page details the API behind this abstraction, which is used quite heavily 
to set up the functionality for defining dynamics models. This should provide the 
insight into the internal workings of the package.

## The `AbstractFunction` Type
```@docs
AbstractFunction
dims
input_dim
```

## Traits on `AbstractFunction`
```@docs
FunctionSignature
DiffMethod
FunctionInputs
functioninputs
```