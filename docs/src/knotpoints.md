# `KnotPoint` type
```@meta
CurrentModule = RobotDynamics
```

A common way of dealing with trajectories of forced dynamical systems, especially in optimization,
is to represent a trajectory with a fixed number of "knot points", typically distributed
evenly over time. Each point records the states, controls, time, and time step to the next
point. It is often convenient to store all this information together, which is the purpose
of the `AbstractKnotPoint` type. Additionally, it is almost always more efficient to index
into a concatenated vector than it is to concatenate two smaller vectors, so the states
and controls are stacked together in a single  `n+m`-dimensional vector.

RobotDynamics.jl defines a couple different implementations of the `AbstractKnotPoint`
interface, which can be useful depending on the application.

## Types
```@docs
AbstractKnotPoint
GeneralKnotPoint
KnotPoint
StaticKnotPoint
```

## Methods
All `AbstractKnotPoint` types support the following methods:

```@docs
state
control
is_terminal
get_z
set_state!
set_control!
set_z!
```

## Mathematical Operations
All `AbstractKnotPoint` types support addition between two knot points, addition of a
knot point and a vector of length `n+m`, and multiplication with a scalar, all of which will
return a `StaticKnotPoint`.
