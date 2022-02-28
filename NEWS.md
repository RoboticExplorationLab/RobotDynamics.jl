# New `v0.4`
Completely new overhaul of the existing API. Now allows for both "inplace" and 
"out-of-place" methods that work with StaticArrays. The method for differentiation 
can be chosen at runtime. No methods are provided by default, but the `@autodiff` macro is 
provided which implements forward automatic differentiation via [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) and finite 
differencing via [FiniteDiff.jl](https://github.com/JuliaDiff/FiniteDiff.jl).

Support for linear models is no longer provided. This functionality is likely to appear in 
a downstream package such as [RobotZoo.jl](https://github.com/RoboticExplorationLab/RobotZoo.jl).
# New in v0.3
## `integrate` method
Allow for more flexibility for defining discrete dynamics by using 
```integrate(Q, model, x, u, t, dt)```
to apply integration rule `Q` instead of `discrete_dynamics`. This allows users to 
customize `discrete_dynamics` for any integration rule `Q`, and optionally apply the 
integration rule to part of the state vector. 

## New options for `state_diff` 
Any valid `Rotations.ErrorMap` to be passed as a 4th argument to `state_diff`, which will be used to compute the error state for all 3D rotations in the `LieState`.

# New in v0.2
## Trajectory Types
Moved trajectory types from TrajectoryOptimization.jl to RobotDynamics.jl.
* `AbstractTrajectory` represents a vector of `AbstractKnotPoints`
* `Traj` is a trajectory of `KnotPoint`s, and has a variety of convenient constructors
* `set_states!`, `set_controls!`, and `get_times!` now live in RobotDynamics.jl

## Trajectory Plotting
Added plotting recipes for plotting trajectories vs time and 2D plots of one state vs another (e.g. `x` and `y` positions)
* `plot(t,X; inds=inds)` plots a trajectory `X` (a vector of static vectors) vs the time `t`. Optional `inds` argument plots a subset of the states.
* `traj2(X; xind=1, yind=2)` plots a 2D trajectory of `x[xind]` vs
`x[yind]`.

## Linear Models (`v0.2.2`)
Created specialized types for dealing with linear models, including `AbstractLinearModel`,
`ContinuousLTI`, `ContinuousLTV`, `DiscreteLTI`, and `DiscreteLTV`. All types can include
an affine term. Custom types that inherit from these abstract types will inherit their
functionality.
