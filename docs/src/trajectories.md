# Trajectories

```@meta
CurrentModule = RobotDynamics
```

When dealing with dynamical sytems, we often need to keep track of and represent 
trajectories of our system. RobotDynamics provides an [`AbstractTrajectory`](@ref)
that can represent any trajectory, continuous or discrete. The only requirements on an 
[`AbstractTrajectory`](@ref) are that you can query the state and control at some time `t`,
and that the trajectory has some notion of initial and final time for which it is defined. 

```@docs
AbstractTrajectory
```

One convenient way of representing a trajectory is by sampling the states and controls 
along it. RobotDynamics provides the `SampledTrajectory` type which is basically a vector 
of [`AbstractKnotPoint`](@ref) types. This is described by the following API:

```@docs
SampledTrajectory
states
controls
gettimes
getdata
setstates!
setcontrols!
set_dt!
setinitialtime!
num_vars
eachcontrol
```

