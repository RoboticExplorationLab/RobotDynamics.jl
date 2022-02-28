# Knot Points

```@meta
CurrentModule = RobotDynamics
```

RobotDynamics describes forced dynamical systems whose behavior is defined by an 
``n``-dimensional state vector ``x`` and an ``m``-dimensional control vector ``u``.
We often represent trajectories of the systems by sampling a continuous trajectory, 
where each sample has a state, control, time, and the time step between the current 
sample and the next. Following terminology from direct trajectory optimization methods
such as direct collocation, we refer to each sample as "knot point." This page 
describes the abstract type `AbstractKnotPoint`, as well a couple different 
implementations of the abstraction provided by the package. These types should be 
sufficient for most use cases: further instantiations of the `AbstractKnotPoint` type 
shouldn't be needed in most cases. 

## The `AbstractKnotPoint` Type
```@docs
AbstractKnotPoint
getstate
getcontrol
state
control
setdata!
setstate!
setcontrol!
is_terminal
vectype
datatype
```

## Concrete knot point types
```@docs
KnotPoint
StaticKnotPoint
```
