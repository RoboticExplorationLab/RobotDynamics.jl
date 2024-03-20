module RobotDynamics

using Rotations
using StaticArrays
using LinearAlgebra
using ForwardDiff
using FiniteDiff
using RecipesBase
using SparseArrays
using Pkg
using Quaternions

using Rotations: skew
using StaticArrays: SUnitRange

include("utils.jl")
include("knotpoint.jl")
include("functionbase.jl")
include("statevectortype.jl")
include("scalar_function.jl")
include("dynamics.jl")
include("discrete_dynamics.jl")

include("jacobian_gen.jl")
include("discretized_dynamics.jl")
include("integration.jl")

include("liestate.jl")
include("rbstate.jl")
include("rigidbody.jl")

include("jacobian.jl")
include("trajectories.jl")
include("plot_recipes.jl")

include("deprecate.jl")

export KnotPoint

const DataVector{T} = Union{Vector{T},StaticVector{<:Any,T},SubArray{T,1}}

using FiniteDiff: compute_epsilon
function FiniteDiff.finite_difference_gradient!(
    df::StridedVector{<:Number},
    f,
    x::StaticVector,
    cache::FiniteDiff.GradientCache{T1,T2,T3,T4,fdtype,returntype,inplace};
    relstep=FiniteDiff.default_relstep(fdtype, eltype(x)),
    absstep=relstep,
    dir=true) where {T1,T2,T3,T4,fdtype,returntype,inplace}

    # c1 is x1 if we need a complex copy of x, otherwise Nothing
    # c2 is Nothing
    fx, c1, c2, c3 = cache.fx, cache.c1, cache.c2, cache.c3
    copyto!(c3, x)
    if fdtype == Val(:forward)
        for i ∈ eachindex(x)
            epsilon = compute_epsilon(fdtype, x[i], relstep, absstep, dir)
            x_old = x[i]
            fx0 = f(x)
            c3[i] += epsilon
            dfi = (f(c3) - fx0) / epsilon
            c3[i] = x_old

            df[i] = real(dfi)
        end
    else
        FiniteDiff.fdtype_error(returntype)
    end
    df
end

end # module
