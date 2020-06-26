```@meta
CurrentModule = RobotDynamics
```

# Models with Rotations

In robotics, the state of our robot often depends on one or more arbitrary 3D rotations
(a.k.a. orientation, attitude). Effectively representing the non-trivial group structure
of rotations has been a topic of study for over 100 years, and as a result many
parameterizations exists. RobotDynamics supports the types defined in
[Rotations.jl](https://github.com/JuliaGeometry/Rotations.jl).

The `LieGroupModel` type allows users to abstract away the particular rotation representation
used and will automatically create efficient methods to handle the potentially different state
dimensions that result. Additionally, it defines methods for operating on the error state,
which for rotations is always three-dimensional. See the discussion in the Rotaitons.jl
[README](https://github.com/JuliaGeometry/Rotations.jl#the-rotation-error-state-and-linearization)
for more information on the error state.


!!! compat
    RobotDynamics requires `v1.0` or higher of `Rotations.jl`


## Defining a `LieGroupModel`
We define a `LieGroupModel` very similarly to that of a standard model. For this example,
let's assume we are modeling a constellation of 2 satellites and we only care about the attitude
dynamics. We will define our state to be `[q1, ω1, q2, ω2]` where `qi` and `ωi` are the
orientation  and angular velocity of the `i`th satellite, respectively.

We start by defining our new type and our dynamics function

```julia
struct SatellitePair{R,T} <: LieGroupModel
    J1::SMatrix{3,3,T,9}   # inertia of satellite 1
    J2::SMatrix{3,3,T,9}   # inertia of satellite 2
end

function RobotDynamics.dynamics(model::SatellitePair, x, u)
    vs = RobotDynamics.vec_states(model, x)  # extract "vector" states
    qs = RobotDynamics.rot_states(model, x)  # extract attitude states
    ω1 = vs[2]  # offset index by 1 since there are now "vector" states before the first quaternion
    ω2 = vs[3]
    q1 = qs[1]
    q2 = qs[2]

    J1, J2 = model.J1, model.J2
    u1 = u[SA[1,2,3]]
    u2 = u[SA[4,5,6]]
    ω1dot = J1\(u1 - ω1 × (J1 * ω1))
    ω2dot = J2\(u2 - ω2 × (J2 * ω2))
    q1dot = Rotations.kinematics(q1, ω1)
    q2dot = Rotations.kinematics(q2, ω2)
    SA[
        q1dot[1], q1dot[2], q1dot[3], q1dot[4],
        ω1dot[1], ω1dot[2], ω1dot[3],
        q2dot[1], q2dot[2], q2dot[3], q2dot[4],
        ω2dot[1], ω2dot[2], ω2dot[3],
    ]
end

RobotDynamics.control_dim(::SatellitePair) = 6
```

Before defining the functions [`vec_states`](@ref) and [`rot_states`](@ref), we will define
the type [`LieState`](@ref), which defines how our state vector is stacked or partitioned.
The `LieState` only needs to know how many "vector" or "non-rotation" states exist, and
where the rotations are placed in the state vector. In our example, we have a rotation,
followed by 3 "vector" states, followed by a rotation, followed by 3 "vector" states, so
we would define our `LieState` to be

```julia
RobotDynamics.LieState(::SatellitePair{R}) where R = RobotDynamics.LieState(R, (0,3,3))
```
which means we have a state with 0 vector states at the beginning, followed by a rotation,
followed by 3 vector states, followed by a rotation, followed by 3 vector states, and the
rotation type is `R`.

With this partitioning in mind, we can now understand the behavior of [`vec_states`](@ref)
and [`rot_states`](@ref), which simply extract the vector and attitude parts of the state
as tuples of `SVector`s.

## `LieGroupModel` API
```@docs
RobotDynamics.LieGroupModel
RobotDynamics.LieState
RobotDynamics.QuatState
RobotDynamics.vec_states
RobotDynamics.rot_states
```
