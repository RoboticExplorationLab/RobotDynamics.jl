using Test
using RobotDynamics
using StaticArrays
using ForwardDiff
using LinearAlgebra
using Random

include("cartpole_model.jl")
include("random_linear.jl")
include("quadrotor.jl")

@testset "Basic Dynamics" begin
    @testset "Cartpole" begin
        include("cartpole_test.jl")
    end
    @testset "Integration" begin
        include("jacobian_test.jl")
        include("integration_tests.jl")
    end
    @testset "KnotPoints" begin
        include("knotpoints.jl")
        include("trajectories.jl")
    end
end

@testset "Lie State" begin
    include("liestate.jl")
    include("liemodel.jl")
end

@testset "Linear Systems" begin
    include("double_integrator.jl")
    include("test_random_linear.jl")
    include("linear_quad.jl")
end

@testset "Rigid Bodies" begin
    @testset "RBState" begin
        include("rbstate.jl")
    end
    @testset "Dynamics" begin
        include("rigidbody.jl")
    end
end
@testset "Jacobians" begin
    include("rigid_body_jacobians.jl")
end

# only test plotting on a desktop since it takes a while to compile Plots on CI...
if !haskey(ENV, "CI")
    @testset "Plotting" begin
        include("plotting.jl")
    end
end