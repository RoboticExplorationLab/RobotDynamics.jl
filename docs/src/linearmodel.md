```@meta
CurrentModule = RobotDynamics
```

# Linear Models
RobotDynamics supports the easy construction of linear models. By defining a linear model, the relevant dynamics
and jacobian functions are predefined for you. This can result in signicant speed ups compared to a naive 
specification of a standard continuous model. These are the types of models currently supported:

* [`AbstractLinearModel`](@ref)
    * [`DiscreteLinearModel`](@ref)
        * [`DiscreteLTI`](@ref)
        * [`DiscreteLTV`](@ref)
    * [`ContinuousLinearModel`](@ref)
        * [`ContinuousLTI`](@ref)
        * [`ContinuousLTV`](@ref)

```@docs
AbstractLinearModel
DiscreteLinearModel
DiscreteLTI
DiscreteLTV
ContinuousLinearModel
ContinuousLTI
ContinuousLTV
get_times
```

# Linearizing a Model

# Discretizing a Model

# Example
