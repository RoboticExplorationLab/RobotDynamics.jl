using RobotDynamics
using Test
using StaticArrays

@testset "Basic Dynamics" begin
    include("cartpole_test.jl")
end

@testset "Lie State" begin
    include("liestate.jl")
end
