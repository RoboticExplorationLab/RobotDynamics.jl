function test_model_state_diff(model)
    n,m = RD.dims(model)
    e = RD.errstate_dim(model)
    x = Vector(rand(model)[1])
    x0 = Vector(rand(model)[1])
    dx = zeros(e)

    RD.state_diff!(model, dx, x, x0)
    @test dx ≈ RD.state_diff(model, x, x0)

    @test (@allocated RD.state_diff!(model, dx, x, x0)) == 0

    # Use static vectors for out-of-place
    x = rand(model)[1]
    x0 = rand(model)[1]
    @test (@allocated RD.state_diff(model, x, x0)) == 0
end

@testset "State Diff ($(typeof(model).name.name))" for model in (Cartpole(), Quadrotor())
    test_model_state_diff(model)
end