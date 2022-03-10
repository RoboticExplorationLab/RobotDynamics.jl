using Test
using RobotDynamics
using StaticArrays
using ForwardDiff
using LinearAlgebra
using FiniteDiff
using Random
using Rotations
const RD = RobotDynamics

include("cartpole_model.jl")
# include("random_linear.jl")
include("quadrotor.jl")

const run_alloc_tests = !haskey(ENV, "CI") 

##
@testset "Function Base" begin
    include("function_base_test.jl")
    include("scalar_function_test.jl")
end

@testset "Basic Dynamics" begin
    @testset "Cartpole" begin
        include("cartpole_test.jl")
    end
    @testset "Integration" begin
        include("integration_tests.jl")
        include("implicit_dynamics_test.jl")
    end
    @testset "KnotPoints" begin
        include("knotpoints.jl")
        include("trajectories.jl")
    end
end

@testset "Lie State" begin
    include("liestate.jl")
    include("liemodel.jl")
    include("state_diff_test.jl")
end

# @testset "Linear Systems" begin
#     include("linmodel_type.jl")
#     include("test_random_linear.jl")
#     include("linear_quad.jl")
#     # include("linear_tests.jl")
#     include("linearization.jl")
# end

@testset "Rigid Bodies" begin
    @testset "RBState" begin
        include("rbstate.jl")
    end
    @testset "Dynamics" begin
        include("rigidbody_test.jl")
    end
end

@testset "Jacobians" begin
    include("jacobian_test.jl")
    include("rigid_body_jacobians.jl")
end

# only test plotting on a desktop since it takes a while to compile Plots on CI
if !haskey(ENV, "CI")
    @testset "Plotting" begin
        include("plotting.jl")
    end
end