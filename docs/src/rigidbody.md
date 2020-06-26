```@meta
CurrentModule = RobotDynamics
```

# Rigid Bodies
Many aerospace systems, such as airplanes, drones, spacecraft, or even underwater vehicles
such as submarines, are conveniently described as a single rigid body described by
its 3D position, orientation, and linear and angular velocities. Since these systems are
relatively common, RobotDynamics provides the `RigidBody` model type that is a special case
of the `LieGroupModel`. In fact, a `RigidBody` is simply a `LieGroupModel` with a `LieState`
of `LieState{R,(3,6)}`, since we define the state vector to be `[r,q,v,ω]` where `r` is
the 3D position, `q` is the orientation of the body in the world frame (`q` rotates vectors
in the body frame to the world frame), `v` is the 3D linear velocity in either the world,
or body frame, and `ω` is the 3D angular velocity in the body frame.

```@docs
RigidBody
```

## Defining a New Rigid Body Model
Let's define the simplest rigid body: a satellite moving freely in 3D space with full 6 DOF
control.

We start by defining a new struct that inherits from `RigidBody{R}` and specifying the number
of controls. Note that `state_dim` should NOT be specified since it is calculated automatically
and depends on the number of parameters in the rotation representation `R`.


```julia
using RobotDynamics
using Rotations
using StaticArrays, LinearAlgebra

# Define the model struct to inherit from `RigidBody{R}`
struct Satellite{R,T} <: RigidBody{R}
    mass::T
    J::Diagonal{T,SVector{3,T}}
end
RobotDynamics.control_dim(::Satellite) = 6
```

We now define a few "getter" methods that are required to evaluation the dynamics.
```julia
RobotDynamics.mass(model::Satellite) = model.mass
RobotDynamics.inertia(model::Satellite) = model.J
```

With those methods specified, all that is left to do is to define the forces and moments
acting on the center of mass of the rigid body. We assume all forces are specified in the
world frame and moments are specified in the body frame.
```julia
# Define the 3D forces at the center of mass, in the world frame
function RobotDynamics.forces(model::Satellite, x::StaticVector, u::StaticVector)
    q = orientation(model, x)
    F = @SVector [u[1], u[2], u[3]]
    q*F  # world frame
end

# Define the 3D moments at the center of mass, in the body frame
function RobotDynamics.moments(model::Satellite, x::StaticVector, u::StaticVector)
    return @SVector [u[4], u[5], u[6]]  # body frame
end
```

Alternatively, we could define the method `wrenches` directly:
```julia
function RobotDynamics.wrenches(model::Satellite, z::AbstractKnotPoint)
    x = state(z)
    u = control(z)
    q = orientation(model, x)
    F = q * (@SVector [u[1], u[2], u[3]])
    M = @SVector [u[4], u[5], u[6]]  # body frame
    return SA[F[1], F[2], F[3], M[1], M[2], M[3]]
end
```
which can take either an `AbstractKnotPoint` or `x` and `u` directly. By passing in a knot
point, we have access to the time `t`, allowing for time-varying wrenches.

### Useful Methods
The following methods are provided to make it easier to define methods for rigid bodies.

To extract the individual components of the state use the following exported methods:
```julia
position(model::RigidBody, x)
orientation(model::RigidBody, x, [renorm=false])
linear_velocity(model::RigidBody, x)
angular_velocity(model::RigidBody, x)
```

Alternatively, you can use the following methods:
```@docs
parse_state
gen_inds
```

Another approach is to use the [`RBState`](@ref), which is useful to describe a generic
state for a rigid body, regardless of the rotation representation:
```@docs
RBState
```


### Building a State Vector
RobotDynamics also provides several convenient methods for building rigid body state vectors.
Identical to `AbstractModel`, `RigidBody` supports `rand` and `zeros`, which uniformly sample
the space of rotations and provide the identity rotation, respectively.

RobotDynamics also provides the following method as a complement to [`parse_state`](@ref) that
builds the state vector from the individual components:
```@docs
build_state
```

## Advanced Usage
RobotDynamics provides a few specialized methods for providing extra performance or customization.

### Specifying the Velocity Frame
Sometimes it is convenient to represent the linear velocity in the body frame instead of the
global frame. This can be changed automatically for our `Satellite` model by defining

```julia
RobotDynamics.velocity_frame(::Satellite) = :body
```
which overrides the default setting of `:world`. Use this same method to query the current
convention, especially in defining the `forces` and `moments` or `wrenches` methods on your
model. Since the forces and moments of our satellite are not functions of the velocity,
we don't need to change anything.

### Faster Continuous Dynamics Jacobians
RobotDynamics has an analytical method for evaluating the continuous dynamics Jacobian
for rigid bodies. If your application only uses the continuous dynamics Jacobian, there
are two ways of getting significant performance improvements:

#### 1. Specify the analytical wrench Jacobian
For our Satellite, we can do this by defining
```julia
function RobotDynamics.wrench_jacobian!(F, model::Satellite, z)
    x = state(z)
    u = control(z)
    q = orientation(model, x)
    ir, iq, iv, iω, iu = RobotDynamics.gen_inds(model)
    iF = SA[1,2,3]
    iM = SA[4,5,6]
    F[iF, iq] .= Rotations.∇rotate(q, u[iF])
    F[iF, iu[iF]] .= RotMatrix(q)
    for i = 1:3
        F[iM[i], iu[i+3]] = 1
    end
    return F
end
```
which gave us about a 60% improvement in runtime.

#### 2. Specify the Wrench Jacobian sparsity
You can get another improvement in performance by overwriting the following method:
```@docs
wrench_sparsity
```
For the satellite we got a 75% improvement in runtime by specifying both the analytical wrench
Jacobian and it's sparsity pattern.
