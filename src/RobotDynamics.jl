module RobotDynamics

using Rotations
using StaticArrays
using LinearAlgebra
using ForwardDiff
using FiniteDiff
using UnsafeArrays
using RecipesBase
using SparseDiffTools
using SparseArrays

using Rotations: skew
using StaticArrays: SUnitRange

# export
#     AbstractModel,
#     DynamicsExpansion,
#     dynamics,
#     jacobian!,
#     discrete_dynamics,
#     discrete_jacobian!,
#     linearize,
#     linearize!,
#     state_dim,
#     control_dim,
#     state_diff_size,
#     rollout!

# # rigid bodies
# export
#     LieGroupModel,
#     RigidBody,
#     RBState,
#     orientation,
#     linear_velocity,
#     angular_velocity

# # linear model
# export
#     LinearModel,
#     linear_dynamics,
#     LinearizedModel,
#     linearize_and_discretize!,
#     discretize,
#     discretize!,
#     update_trajectory!

# # knotpoints
# export
#     AbstractKnotPoint,
#     KnotPoint,
#     StaticKnotPoint,
#     Traj,
#     state,
#     control,
#     states,
#     controls,
#     set_states!,
#     set_controls!

# # integration
# export
#     QuadratureRule,
#     RK2,
#     RK3,
#     RK4,
#     HermiteSimpson,
#     PassThrough,
#     Exponential


# include("rbstate.jl")
# include("jacobian.jl")
# include("knotpoint.jl")
# include("model.jl")
# include("liestate.jl")
# include("rigidbody.jl")
# include("integration.jl")
# include("trajectories.jl")
# include("linearmodel.jl")
# include("linearization.jl")
# include("plot_recipes.jl")

include("knotpoint.jl")
include("functionbase.jl")
include("jacobian_gen.jl")
include("dynamics.jl")
include("discrete_dynamics.jl")

end # module
