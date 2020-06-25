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
    @testset "KnotPoints" begin
        include("knotpoints.jl")
    end
end

@testset "Lie State" begin
    include("liestate.jl")
    include("liemodel.jl")
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
