[![Build Status](https://travis-ci.com/RoboticExplorationLab/RobotDynamics.jl.svg?branch=master)](https://travis-ci.com/RoboticExplorationLab/RobotDynamics.jl)
![CI](https://github.com/RoboticExplorationLab/RobotDynamics.jl/workflows/CI/badge.svg)
[![codecov](https://codecov.io/gh/RoboticExplorationLab/RobotDynamics.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/RoboticExplorationLab/RobotDynamics.jl)

# RobotDynamics.jl

This purpose of this package is to provide a common interface for calling systems with
forced dynamics, i.e. dynamics of the form `xdot = f(x,u,t)` where `x` is an n-dimensional
state vector and `u` is an m-dimensional control vector.

A model is defined by creating a custom type that inherits from `AbstractModel` and then
defining a custom `Dynamics.dynamics` function on your model, along with `Base.size` that
gives n and m. With this information, `Dynamics.jl` provides methods for computing discrete
dynamics as well as continuous and discrete time dynamics Jacobians.

This package also supports dynamics of rigid bodies, whose state is defined by a position,
orientation, and linear and angular velocities. This package provides methods that correctly
compute the dynamics Jacobians for the 3D attitude representions implemented in
`DifferentialRotations.jl`.

Internally, this package makes use of the `Knotpoint` type that is a custom type that simply
stores the state and control vectors along with the current time and time step length all
together in a single type.
