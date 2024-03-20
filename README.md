![CI](https://github.com/RoboticExplorationLab/RobotDynamics.jl/workflows/CI/badge.svg)
[![codecov](https://codecov.io/gh/RoboticExplorationLab/RobotDynamics.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/RoboticExplorationLab/RobotDynamics.jl)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://RoboticExplorationLab.github.io/RobotDynamics.jl/dev)

# RobotDynamics.jl

This purpose of this package is to provide a common interface for calling systems with
forced dynamics, i.e. dynamics of the form `xdot = f(x,u,t)` where `x` is an n-dimensional
state vector and `u` is an m-dimensional control vector.

A model is defined by creating a custom type that inherits from `AbstractModel` and then
defining a custom `dynamics` function on your model, along with `state_dim` and `control_dim` that
gives n and m. With this information, `RobotDynamics.jl` provides methods for computing discrete
dynamics as well as continuous and discrete time dynamics Jacobians.

This package also supports dynamics of rigid bodies and models whose state depends on one or more arbitrary 3D rotations. 
This package provides methods that correctly
compute the dynamics Jacobians for the 3D attitude representions implemented in
`Rotations.jl`.

Internally, this package makes use of the `Knotpoint` type that is a custom type that simply
stores the state and control vectors along with the current time and time step length all
together in a single type.

## Basic Example: Cartpole
```julia
using RobotDynamics
using StaticArrays

# Define the model struct with parameters
struct Cartpole{T} <: AbstractModel
    mc::T
    mp::T
    l::T
    g::T
end

Cartpole() = Cartpole(1.0, 0.2, 0.5, 9.81)

# Define the continuous dynamics
function RobotDynamics.dynamics(model::Cartpole, x, u)
    mc = model.mc  # mass of the cart in kg (10)
    mp = model.mp   # mass of the pole (point mass at the end) in kg
    l = model.l   # length of the pole in m
    g = model.g  # gravity m/s^2

    q = x[ @SVector [1,2] ]
    qd = x[ @SVector [3,4] ]

    s = sin(q[2])
    c = cos(q[2])

    H = @SMatrix [mc+mp mp*l*c; mp*l*c mp*l^2]
    C = @SMatrix [0 -mp*qd[2]*l*s; 0 0]
    G = @SVector [0, mp*g*l*s]
    B = @SVector [1, 0]

    qdd = -H\(C*qd + G - B*u[1])
    return [qd; qdd]
end

# Specify the state and control dimensions
RobotDynamics.state_dim(::Cartpole) = 4
RobotDynamics.control_dim(::Cartpole) = 1

# Create the model
model = Cartpole()
n,m = RD.dims(model)

# Generate random state and control vector
x,u = rand(model)
dt = 0.1  # time step (s)
z = KnotPoint(x,u,dt)

# Evaluate the continuous dynamics and Jacobian
ẋ = dynamics(model, x, u)
∇f = RobotDynamics.DynamicsJacobian(model)
jacobian!(∇f, model, z)

# Evaluate the discrete dynamics and Jacobian
x′ = discrete_dynamics(RK3, model, z)
discrete_jacobian!(RK3, ∇f, model, z)
```


## Rigid Body Example: Satellite
```julia
using RobotDynamics
using Rotations
using StaticArrays, LinearAlgebra
using BenchmarkTools

# Define the model struct to inherit from `RigidBody{R}`
struct Satellite{R,T} <: RigidBody{R}
    mass::T
    J::Diagonal{T,SVector{3,T}}
end
RobotDynamics.control_dim(::Satellite) = 6

# Define some simple "getter" methods that are required to evaluate the dynamics
RobotDynamics.mass(model::Satellite) = model.mass
RobotDynamics.inertia(model::Satellite) = model.J

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


# Build model
T = Float64
R = QuatRotation{T}
mass = 1.0
J = Diagonal(@SVector ones(3))
model = Satellite{R,T}(mass, J)

# Initialization
x,u = rand(model)   # generate a state with the rotation uniformly sampled from the space of rotations
dt = 0.1            # time step (s)
z = KnotPoint(x,u,dt)
∇f = RobotDynamics.DynamicsJacobian(model)

# Continuous dynamics
dynamics(model, x, u)
jacobian!(∇f, model, z)

# Discrete dynamics
discrete_dynamics(RK2, model, z)
discrete_jacobian!(RK2, ∇f, model, z)
```
