using Test
using RobotDynamics
using StaticArrays
using ForwardDiff

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


@testset "Rigid Bodies" begin
    @testset "Lie State" begin
        include("liestate.jl")
    end

    @testset "Single Rigid Body" begin
        include("rbstate.jl")
        include("rigidbody.jl")
    end
end
