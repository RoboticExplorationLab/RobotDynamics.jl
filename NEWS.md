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