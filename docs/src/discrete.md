# Discrete Dynamics

```@meta
CurrentModule = RobotDynamics
```

This page provides the API for discrete dynamics models.

```@contents
Pages = ["discrete.md"]
Depth = 3
```

## Discrete Dynamics type
```@docs
DiscreteDynamics
```

## Methods for Discrete Dynamics models
```@docs
discrete_dynamics
discrete_dynamics!
dynamics_error
dynamics_error!
dynamics_error_jacobian!
propagate_dynamics!
```

## Discretizing Continuous Models

```@docs
DiscretizedDynamics
QuadratureRule
Explicit
Implicit
```

## Implemented Integrators
### Explicit
```@docs
Euler
RK3
RK4
```

### Implicit
```@docs
ImplicitMidpoint
```

### Interal API
```@docs
ADVector
```
