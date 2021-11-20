module RobotDynamics

using Rotations
using StaticArrays
using LinearAlgebra
using ForwardDiff
using FiniteDiff
using RecipesBase
using SparseArrays

using Rotations: skew
using StaticArrays: SUnitRange

include("utils.jl")
include("knotpoint.jl")
include("functionbase.jl")
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
    if fdtype != Val(:complex)
        if eltype(df)<:Complex && !(eltype(x)<:Complex)
            copyto!(c1,x)
        end
    end
    copyto!(c3,x)
    if fdtype == Val(:forward)
        for i ∈ eachindex(x)
            epsilon = compute_epsilon(fdtype, x[i], relstep, absstep, dir)
            x_old = x[i]
            if typeof(fx) != Nothing
                c3[i] += epsilon
                dfi = (f(c3) - fx) / epsilon
                c3[i] = x_old
            else
                fx0 = f(x)
                c3[i] += epsilon
                dfi = (f(c3) - fx0) / epsilon
                c3[i] = x_old
            end

            df[i] = real(dfi)
            if eltype(df)<:Complex
                if eltype(x)<:Complex
                    c3[i] += im * epsilon
                    if typeof(fx) != Nothing
                        dfi = (f(c3) - fx) / (im*epsilon)
                    else
                        dfi = (f(c3) - fx0) / (im*epsilon)
                    end
                    c3[i] = x_old
                else
                    c1[i] += im * epsilon
                    if typeof(fx) != Nothing
                        dfi = (f(c1) - fx) / (im*epsilon)
                    else
                        dfi = (f(c1) - fx0) / (im*epsilon)
                    end
                    c1[i] = x_old
                end
                df[i] -= im * imag(dfi)
            end
        end
    elseif fdtype == Val(:central)
        @inbounds for i ∈ eachindex(x)
            epsilon = compute_epsilon(fdtype, x[i], relstep, absstep, dir)
            x_old = x[i]
            c3[i] += epsilon
            dfi = f(c3)
            c3[i] = x_old - epsilon
            dfi -= f(c3)
            c3[i] = x_old
            df[i] = real(dfi / (2*epsilon))
            if eltype(df)<:Complex
                if eltype(x)<:Complex
                    c3[i] += im*epsilon
                    dfi = f(c3)
                    c3[i] = x_old - im*epsilon
                    dfi -= f(c3)
                    c3[i] = x_old
                else
                    c1[i] += im*epsilon
                    dfi = f(c1)
                    c1[i] = x_old - im*epsilon
                    dfi -= f(c1)
                    c1[i] = x_old
                end
                df[i] -= im*imag(dfi / (2*im*epsilon))
            end
        end
    elseif fdtype==Val(:complex) && returntype<:Real && eltype(df)<:Real && eltype(x)<:Real
        copyto!(c1,x)
        epsilon_complex = eps(real(eltype(x)))
        # we use c1 here to avoid typing issues with x
        @inbounds for i ∈ eachindex(x)
            c1_old = c1[i]
            c1[i] += im*epsilon_complex
            df[i]  = imag(f(c1)) / epsilon_complex
            c1[i]  = c1_old
        end
    else
        fdtype_error(returntype)
    end
    df
end

end # module
