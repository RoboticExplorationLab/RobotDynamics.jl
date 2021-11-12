using RobotDynamics
using StaticArrays
using Test
using BenchmarkTools
using ForwardDiff, FiniteDiff

using RobotDynamics: @autodiff, KnotPoint, getdata,
    StaticReturn, InPlace, ForwardAD, FiniteDifference
const RD = RobotDynamics

function test_model(model)
    n,m = size(model)
    xdot = zeros(n)
    x = @SVector randn(n)
    u = @SVector randn(m)
    t,dt = 0.0, 0.1
    zs = KnotPoint(x,u,t,dt)
    z = KnotPoint(Vector(x),Vector(u),t,dt)
    J = zeros(n, n + m)
    J0 = zeros(n, n + m)

    RD.evaluate!(model, xdot, z)
    @test xdot ≈ RD.evaluate(model, z)

    RD.evaluate!(model, xdot, x, u, RD.getparams(z))
    @test xdot ≈ RD.evaluate(model, x, u, RD.getparams(z))

    if model isa RD.ContinuousDynamics
        RobotDynamics.dynamics!(model, xdot, x, u, t)
        @test xdot ≈ RobotDynamics.dynamics(model, x, u, t) 
    elseif model isa RD.DiscreteDynamics
        xn = xdot
        RobotDynamics.discrete_dynamics!(model, xn, x, u, t, dt)
        @test xn ≈ RobotDynamics.discrete_dynamics(model, x, u, t, dt)
    end

    RobotDynamics.jacobian!(StaticReturn(), RD.UserDefined(), model, J0, xdot, z)
    RobotDynamics.jacobian!(StaticReturn(), ForwardAD(), model, J, xdot, z)
    @test J ≈ J0
    RobotDynamics.jacobian!(InPlace(), RD.UserDefined(), model, J0, xdot, z)
    RobotDynamics.jacobian!(InPlace(), ForwardAD(), model, J, xdot, z)
    @test J ≈ J0
    RobotDynamics.jacobian!(StaticReturn(), FiniteDifference(), model, J, xdot, z)
    @test J ≈ J0 atol = 1e-6
    RobotDynamics.jacobian!(InPlace(), FiniteDifference(), model, J, xdot, z)
    @test J ≈ J0 atol = 1e-6

    allocs = 0
    if model isa RD.ContinuousDynamics
        allocs += @allocated RobotDynamics.dynamics!(model, xdot, x, u, t)
        allocs += @allocated RobotDynamics.dynamics(model, x, u, t) 
    else
        allocs += @allocated RobotDynamics.discrete_dynamics!(model, xdot, x, u, t, dt)
        allocs += @allocated RobotDynamics.discrete_dynamics(model, x, u, t, dt) 
    end
    allocs += @allocated RobotDynamics.evaluate!(model, xdot, z)
    allocs += @allocated RobotDynamics.evaluate(model, zs)
    allocs += @allocated RobotDynamics.evaluate!(model, xdot, x, u, RD.getparams(z))
    allocs += @allocated RobotDynamics.evaluate(model, x, u, RD.getparams(z))
    allocs += @allocated RobotDynamics.jacobian!(StaticReturn(), RD.UserDefined(), model, J, xdot, zs)
    allocs += @allocated RobotDynamics.jacobian!(InPlace(), RD.UserDefined(), model, J, xdot, zs)
    allocs += @allocated RobotDynamics.jacobian!(StaticReturn(), ForwardAD(), model, J, xdot, zs)
    allocs += @allocated RobotDynamics.jacobian!(InPlace(), ForwardAD(), model, J, xdot, z)
    allocs += @allocated RobotDynamics.jacobian!(StaticReturn(), FiniteDifference(), model, J, xdot, zs)
    allocs += @allocated RobotDynamics.jacobian!(InPlace(), FiniteDifference(), model, J, xdot, z)
    @test allocs == 0
end

function test_error_allocs(model)
    n,m = size(model)
    z1 = KnotPoint(randn(model)..., 0.0, 0.1)
    z2 = KnotPoint(randn(model)..., 0.1, 0.1)
    J1 = zeros(n, n+m)
    J2 = copy(J1)
    y1 = zeros(n)
    y2 = copy(y1)
    RD.dynamics_error!(model, y2, y1, z2, z1)
    @test y2 ≈ RD.dynamics_error(model, z2, z1)

    allocs = 0
    allocs += @allocated RD.dynamics_error_jacobian!(StaticReturn(), RD.UserDefined(), model, J2, J1, y2, y1, z2, z1)
    allocs += @allocated RD.dynamics_error_jacobian!(InPlace(), RD.UserDefined(), model, J2, J1, y2, y1, z2, z1)
    allocs += @allocated RD.dynamics_error_jacobian!(StaticReturn(), ForwardAD(), model, J2, J1, y2, y1, z2, z1)
    allocs += @allocated RD.dynamics_error_jacobian!(InPlace(), ForwardAD(), model, J2, J1, y2, y1, z2, z1)
    allocs += @allocated RD.dynamics_error_jacobian!(StaticReturn(), FiniteDifference(), model, J2, J1, y2, y1, z2, z1)
    allocs += @allocated RD.dynamics_error_jacobian!(InPlace(), FiniteDifference(), model, J2, J1, y2, y1, z2, z1)

    z1 = KnotPoint{n,m}(Vector(z1.z), z1.t, z1.dt)
    z2 = KnotPoint{n,m}(Vector(z2.z), z2.t, z2.dt)
    allocs += @allocated RD.dynamics_error_jacobian!(InPlace(), RD.UserDefined(), model, J2, J1, y2, y1, z2, z1)
    allocs += @allocated RD.dynamics_error_jacobian!(InPlace(), ForwardAD(), model, J2, J1, y2, y1, z2, z1)
    allocs += @allocated RD.dynamics_error_jacobian!(InPlace(), FiniteDifference(), model, J2, J1, y2, y1, z2, z1)
    @test allocs == 0
end

## 
abstract type DI{D} <: RobotDynamics.ContinuousDynamics where {D} end
RobotDynamics.state_dim(::DI{D}) where {D} = 2D
RobotDynamics.control_dim(::DI{D}) where {D} = D

@generated function RobotDynamics.dynamics(fun::DI{D}, x, u, t) where {D}
    N, M = 2D, D
    vel = [:(x[$i]) for i = M+1:N]
    us = [:(u[$i]) for i = 1:M]
    :(SVector{$N}($(vel...), $(us...)))
end

@generated function RobotDynamics.dynamics!(fun::DI{D}, y, x, u, t) where {D}
    N, M = 2D, D
    vel = [:(y[$(i - M)] = x[$i]) for i = M+1:N]
    us = [:(y[$(i + M)] = u[$i]) for i = 1:M]
    quote
        $(Expr(:block, vel...))
        $(Expr(:block, us...))
        return nothing
    end
end

@generated function RobotDynamics.jacobian!(fun::DI{D}, J, y, x, u, t) where {D}
    N, M = 2D, D
    jac = [:(J[$i, $(i + M)] = 1.0) for i = 1:N]
    quote
        J .= 0
        $(Expr(:block, jac...))
        return nothing
    end
end

