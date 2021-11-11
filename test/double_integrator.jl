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

    RobotDynamics.evaluate!(model, xdot, z)
    @test xdot ≈ RobotDynamics.evaluate(model, z)
    RobotDynamics.dynamics!(model, xdot, x, u, t)
    @test xdot ≈ RobotDynamics.dynamics(model, x, u, t) 

    RobotDynamics.jacobian!(model, J0, xdot, z)
    RobotDynamics.jacobian!(StaticReturn(), ForwardAD(), model, J, xdot, z)
    @test J ≈ J0
    RobotDynamics.jacobian!(InPlace(), ForwardAD(), model, J, xdot, z)
    @test J ≈ J0
    RobotDynamics.jacobian!(StaticReturn(), FiniteDifference(), model, J, xdot, z)
    @test J ≈ J0 atol = 1e-6
    RobotDynamics.jacobian!(InPlace(), FiniteDifference(), model, J, xdot, z)
    @test J ≈ J0 atol = 1e-6

    allocs = 0
    allocs += @allocated RobotDynamics.evaluate!(model, xdot, z)
    allocs += @allocated RobotDynamics.evaluate(model, zs)
    allocs += @allocated RobotDynamics.dynamics!(model, xdot, x, u, t)
    allocs += @allocated RobotDynamics.dynamics(model, x, u, t) 
    allocs += @allocated RobotDynamics.jacobian!(model, J0, xdot, z)
    allocs += @allocated RobotDynamics.jacobian!(StaticReturn(), ForwardAD(), model, J, xdot, zs)
    allocs += @allocated RobotDynamics.jacobian!(InPlace(), ForwardAD(), model, J, xdot, z)
    allocs += @allocated RobotDynamics.jacobian!(StaticReturn(), FiniteDifference(), model, J, xdot, zs)
    allocs += @allocated RobotDynamics.jacobian!(InPlace(), FiniteDifference(), model, J, xdot, z)
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
        $(Expr(:block, jac...))
        return nothing
    end
end

## Example
mutable struct DoubleIntegrator{D,NM} <: DI{D}
    cfg::ForwardDiff.JacobianConfig{
        Nothing,
        Float64,
        NM,
        Tuple{
            Vector{ForwardDiff.Dual{Nothing,Float64,NM}},
            Vector{ForwardDiff.Dual{Nothing,Float64,NM}},
        },
    }
    cache::FiniteDiff.JacobianCache{
        Vector{Float64},
        Vector{Float64},
        Vector{Float64},
        UnitRange{Int64},
        Nothing,
        Val{:forward}(),
        Float64,
    }
    function DoubleIntegrator{D}() where {D}
        n = 2D
        m = D
        y = zeros(n)
        z = zeros(n + m)
        cfg = ForwardDiff.JacobianConfig(nothing, y, z)
        cache = FiniteDiff.JacobianCache(z, y)
        new{D,length(cfg.seeds)}(cfg, cache)
    end
end

function RobotDynamics.jacobian!(::StaticReturn, ::ForwardAD, model::DoubleIntegrator, J, y, z)
    f(_z) = RD.evaluate(model, RD.getstate(z, _z), RD.getcontrol(z, _z), RD.getparams(z))
    J .= ForwardDiff.jacobian(f, getdata(z))
end

function RobotDynamics.jacobian!(::InPlace, ::ForwardAD, model::DoubleIntegrator, J, y, z)
    f!(_y, _z) = RD.evaluate!(model, _y, RD.getstate(z, _z), RD.getcontrol(z, _z), RD.getparams(z))
    J .= ForwardDiff.jacobian!(J, f!, y, getdata(z), model.cfg)
end

function RobotDynamics.jacobian!(::StaticReturn, ::FiniteDifference, model::DoubleIntegrator, J, y, z)
    f!(_y,_z) = _y .= RD.evaluate(model, RD.getstate(z, _z), RD.getcontrol(z, _z), RD.getparams(z))
    FiniteDiff.finite_difference_jacobian!(J, f!, RD.getdata(z), model.cache)
end

function RobotDynamics.jacobian!(::InPlace, ::FiniteDifference, model::DoubleIntegrator, J, y, z)
    f!(_y,_z) = RD.evaluate!(model, _y, RD.getstate(z, _z), RD.getcontrol(z, _z), RD.getparams(z))
    FiniteDiff.finite_difference_jacobian!(J, f!, RD.getdata(z), model.cache)
end

dim = 3
model = DoubleIntegrator{dim}()
@test RobotDynamics.state_dim(model) == 2dim
@test RobotDynamics.control_dim(model) == dim
@test RobotDynamics.output_dim(model) == 2dim
test_model(model)

## Model with auto-generated jacobian methods
@autodiff struct DoubleIntegratorAuto{D} <: DI{D} end
model = DoubleIntegratorAuto{dim}()
test_model(model)