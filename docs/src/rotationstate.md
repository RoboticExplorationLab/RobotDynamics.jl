# The Rotation State
```@meta
CurrentModule = RobotDynamics
```
As mentioned in [AbstractFunction](@ref AbstractFunction), RobotDynamics can work
with non-Euclidean state vectors, or state vectors whose composition rule is not 
simple addition. The most common example of non-Euclidean state vectors in robotics 
is that of 3D rotations. Frequently our state vectors include a 3D rotation together 
with normal Euclidean states such as position, linear or angular velocities, etc. 
The [`RotationState`](@ref) [`StateVectorType`](@ref) represents this type of state vector.
In general, this represents a state vector of the following form:

```math
\begin{bmatrix}
x_1 \\
q_1 \\
x_2 \\
q_2 \\
\vdots \\
x_{N-1} \\
q_N \\
x_N
\end{bmatrix}
```
where ``x_k \in \mathbb{R}^{n_k}`` and ``q_k \in SO(3)``. Any of the ``n_k`` can be zero.

This state is described by the [`LieState`](@ref) struct:
```@docs
LieState
QuatState
```

## The `LieGroupModel` type
To simplify the definition of models whose state vector is a [`RotationState`](@ref), we 
provide the abstract [`LieGroupModel`](@ref) type:

```@docs
LieGroupModel
```

## Single rigid bodies
A lot of robotic systems, such as airplanes, quadrotors, underwater vehicles, satellites, 
etc., can be described as a single rigid body subject to external forces and actuators.
RobotDynamics provides the [`RigidBody`](@ref) model type for this type of system:

```@docs
RigidBody
Base.position(::RBState)
orientation
linear_velocity
angular_velocity
build_state
parse_state
gen_inds
flipquat
```

## The `RBState` type
When working with rigid bodies, the rotation can be represented a variety of methods and 
dealing with this ambiguity can be tedious. We provide the [`RBState`](@ref) type which 
represents a generic state for a rigid body, representing the orietation as a quaternion.
It provides easy methods to convert to and from the state vector for a given `RigidBody{R}`.

```@docs
RBState
```