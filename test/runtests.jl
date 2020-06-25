using Test
using RobotDynamics
using StaticArrays
using ForwardDiff
using LinearAlgebra

include("cartpole_model.jl")

@testset "Basic Dynamics" begin
    @testset "Cartpole" begin
        include("cartpole_test.jl")
    end
    @testset "Integration" begin
        include("jacobian_test.jl")
        include("integration_tests.jl")
    end
end

@testset "Lie State" begin
    include("liestate.jl")
end

@testset "Rigid Bodies" begin
    @testset "RBState" begin
        include("rbstate.jl")
    end
    @testset "Dynamics" begin
        include("rigidbody.jl")
    end
    @testset "Jacobians"
        include("rigid_body_jacobians.jl")
    end
end
