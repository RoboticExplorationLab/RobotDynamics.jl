using RobotDynamics
using Test
using StaticArrays

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

    @testset "RBState" begin
        include("rbstate.jl")
        include("rigidbody.jl")
    end
end
