include("double_integrator.jl")

## Double Integrator Example
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

# These methods are automatically defined by @autodiff
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