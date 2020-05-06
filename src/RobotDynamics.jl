module RobotDynamics

using Rotations
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
    linearize,
    linearize!,
    state_dim,
    control_dim,
    state_diff_size,
    rollout!

# rigid bodies
export
    LieGroupModel,
    RigidBody,
    RBState,
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
    QuadratureRule,
    RK2,
    RK3,
    RK4,
    HermiteSimpson


include("knotpoint.jl")
include("model.jl")
include("rigidbody.jl")
include("integration.jl")
include("trajectories.jl")
include("rbstate.jl")
include("liestate.jl")

end # module
