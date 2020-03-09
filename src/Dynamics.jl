module Dynamics

using DocStringExtensions
using DifferentialRotations
using StaticArrays
using LinearAlgebra
using ForwardDiff

export
    AbstractModel,
    DynamicsExpansion,
    dynamics,
    jacobian!,
    discrete_dynamics,
    discrete_jacobian!,
    state_dim,
    control_dim,
    state_diff_size,
    rollout!

# rigid bodies
export
    RigidBody,
    orientation,
    linear_velocity,
    angular_velocity


# knotpoints
export
    AbstractKnotPoint,
    KnotPoint,
    StaticKnotPoint,
    Traj,
    state,
    control,
    states,
    controls,
    set_states!,
    set_controls!

# integration
export
    RK2,
    RK3,
    RK4,
    HermiteSimpson


include("expansion.jl")
include("knotpoint.jl")
include("model.jl")
include("rigidbody.jl")
include("integration.jl")
include("trajectories.jl")

end # module
