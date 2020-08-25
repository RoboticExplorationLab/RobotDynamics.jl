module RobotDynamics

using Rotations
using StaticArrays
using LinearAlgebra
using ForwardDiff
using UnsafeArrays
using RecipesBase

using Rotations: skew
using StaticArrays: SUnitRange

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

# linear model
export
    AbstractLinearModel,
    DiscreteLinearModel,
    DiscreteLTV,
    DiscreteLTI,
    DiscreteSystemQuadrature,
    ContinuousLinearModel,
    ContinuousLTV,
    ContinuousLTI,
    get_A,
    get_B,
    get_d,
    set_A!,
    set_B!,
    set_d!,
    is_affine,
    is_time_varying,
    linearize_and_discretize!,
    discretize!,
    Exponential,
    Euler,
    @create_continuous_lti,
    @create_discrete_lti,
    @create_continuous_ltv,
    @create_discrete_ltv


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


include("rbstate.jl")
include("jacobian.jl")
include("knotpoint.jl")
include("model.jl")
include("liestate.jl")
include("rigidbody.jl")
include("integration.jl")
include("trajectories.jl")
include("linearmodel.jl")
include("linearization.jl")
include("plot_recipes.jl")

end # module
