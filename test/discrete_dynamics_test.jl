include("double_integrator.jl")

## Discrete Double Integrator model
struct DiscreteModel{NM} <: RobotDynamics.DiscreteDynamics 
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
    function DiscreteModel()
        n,m = 4,2
        y = zeros(n)
        z = zeros(n + m)
        cfg = ForwardDiff.JacobianConfig(nothing, y, z)
        cache = FiniteDiff.JacobianCache(z, y)
        new{length(cfg.seeds)}(cfg, cache)
    end
end
RD.state_dim(::DiscreteModel) = 4
RD.control_dim(::DiscreteModel) = 2

function RD.discrete_dynamics(model::DiscreteModel, x, u, t, dt)
    SA[
        x[1] + x[3] * dt,
        x[2] + x[4] * dt,
        x[3] + u[1] * dt,
        x[4] + u[2] * dt,
    ]
end

function RD.discrete_dynamics!(model::DiscreteModel, xn, x, u, t, dt)
    xn[1] = x[1] + x[3] * dt
    xn[2] = x[2] + x[4] * dt
    xn[3] = x[3] + u[1] * dt
    xn[4] = x[4] + u[2] * dt
    return nothing
end

function RD.jacobian!(model::DiscreteModel, J, xn, x, u, t, dt)
    J .= 0
    J[1,1] = 1
    J[1,3] = dt
    J[2,2] = 1
    J[2,4] = dt
    J[3,3] = 1
    J[3,5] = dt
    J[4,4] = 1
    J[4,6] = dt
    return nothing
end

function RD.dynamics_error_jacobian!(model::DiscreteModel, J2, J1, y2, y1, z2::RD.AbstractKnotPoint, z1::RD.AbstractKnotPoint)
    n = RD.state_dim(model)
    RD.jacobian!(model, J1, y2, z1)
    J2 .= 0
    for i = 1:n 
        J2[i,i] = -1.0
    end
    return nothing
end

# Explicit Dynamics derivatives
#   These are automatically defined by @autodiff
function RobotDynamics.jacobian!(::StaticReturn, ::ForwardAD, model::DiscreteModel, J, y, z)
    f(_z) = RD.evaluate(model, RD.getstate(z, _z), RD.getcontrol(z, _z), RD.getparams(z))
    J .= ForwardDiff.jacobian(f, getdata(z))
end

function RobotDynamics.jacobian!(::InPlace, ::ForwardAD, model::DiscreteModel, J, y, z)
    f!(_y, _z) = RD.evaluate!(model, _y, RD.getstate(z, _z), RD.getcontrol(z, _z), RD.getparams(z))
    J .= ForwardDiff.jacobian!(J, f!, y, getdata(z), model.cfg)
end

function RobotDynamics.jacobian!(::StaticReturn, ::FiniteDifference, model::DiscreteModel, J, y, z)
    f!(_y,_z) = _y .= RD.evaluate(model, RD.getstate(z, _z), RD.getcontrol(z, _z), RD.getparams(z))
    FiniteDiff.finite_difference_jacobian!(J, f!, RD.getdata(z), model.cache)
end

function RobotDynamics.jacobian!(::InPlace, ::FiniteDifference, model::DiscreteModel, J, y, z)
    f!(_y,_z) = RD.evaluate!(model, _y, RD.getstate(z, _z), RD.getcontrol(z, _z), RD.getparams(z))
    FiniteDiff.finite_difference_jacobian!(J, f!, RD.getdata(z), model.cache)
end

# Dynamics error (implicit dynamics) derivatives
#   These are automatically defined by @autodiff for DiscreteDynamics models
function RD.dynamics_error_jacobian!(::StaticReturn, ::ForwardAD, model::DiscreteModel, J2, J1, y2, y1, z2::KnotPoint, z1::KnotPoint)
    f1(_z) = RD.dynamics_error(model, z2, RD.StaticKnotPoint(z1, _z))
    J1 .= ForwardDiff.jacobian(f1, RD.getdata(z1))

    f2(_z) = RD.dynamics_error(model, RD.StaticKnotPoint(z2, _z), z1)
    J2 .= ForwardDiff.jacobian(f2, RD.getdata(z2))
    return nothing
end

function RD.dynamics_error_jacobian!(::InPlace, ::ForwardAD, model::DiscreteModel, J2, J1, y2, y1, z2::KnotPoint, z1::KnotPoint)
    f1!(_y, _z) = RD.dynamics_error!(model, _y, y1, z2, RD.StaticKnotPoint(z1, _z))
    ForwardDiff.jacobian!(J1, f1!, y2, RD.getdata(z1), model.cfg)

    f2!(_y, _z) = RD.dynamics_error!(model, _y, y1, RD.StaticKnotPoint(z2, _z), z1)
    ForwardDiff.jacobian!(J1, f1!, y2, RD.getdata(z2), model.cfg)
    return nothing
end

function RD.dynamics_error_jacobian!(::StaticReturn, ::FiniteDifference, model::DiscreteModel, J2, J1, y2, y1, z2::KnotPoint, z1::KnotPoint)
    f1!(_y, _z) = _y .= RD.dynamics_error(model, z2, RD.StaticKnotPoint(z1, _z))
    FiniteDiff.finite_difference_jacobian!(J1, f1!, RD.getdata(z1), model.cache)

    f2!(_y, _z) = _y .= RD.dynamics_error(model, RD.StaticKnotPoint(z2, _z), z1)
    FiniteDiff.finite_difference_jacobian!(J2, f2!, RD.getdata(z2), model.cache)
    return nothing
end

function RD.dynamics_error_jacobian!(::InPlace, ::FiniteDifference, model::DiscreteModel, J2, J1, y2, y1, z2::KnotPoint, z1::KnotPoint)
    f1!(_y, _z) = RD.dynamics_error!(model, _y, y1, z2, RD.StaticKnotPoint(z1, _z))
    FiniteDiff.finite_difference_jacobian!(J1, f1!, RD.getdata(z1), model.cache)

    f2!(_y, _z) = RD.dynamics_error!(model, _y, y1, RD.StaticKnotPoint(z2, _z), z1)
    FiniteDiff.finite_difference_jacobian!(J2, f2!, RD.getdata(z2), model.cache)
    return nothing
end

model = DiscreteModel()
test_model(model)

# Test dynamics error
z1 = KnotPoint(randn(model)..., 0, 0.1)
x2 = RD.evaluate(model, z1) 
z2 = KnotPoint(x2, RD.control(z1)*0, 0.1, z1.dt)
RD.control(z1)
@test RD.dynamics_error(model, z2, z1) == zeros(4)

# Test dynamics error jacobian
n,m = size(model)
J1 = zeros(n, n+m); J2 = copy(J1)
y1 = zeros(n);      y2 = copy(y1)
J10 = copy(J1);     J20 = copy(J2)
RD.dynamics_error_jacobian!(model, J20, J10, y2, y1, z2, z1)
RD.dynamics_error_jacobian!(StaticReturn(), ForwardAD(), model, J2, J1, y2, y1, z2, z1)
@test J10 ≈ J1
@test J20 ≈ J2
RD.dynamics_error_jacobian!(StaticReturn(), RD.UserDefined(), model, J2, J1, y2, y1, z2, z1)
@test J10 ≈ J1
@test J20 ≈ J2

test_error_allocs(model)

## Discrete model w/ autogen
@autodiff struct DiscreteModelAuto <: RobotDynamics.DiscreteDynamics end
RD.state_dim(::DiscreteModelAuto) = 4
RD.control_dim(::DiscreteModelAuto) = 2

function RD.discrete_dynamics(model::DiscreteModelAuto, x, u, t, dt)
    SA[
        x[1] + x[3] * dt,
        x[2] + x[4] * dt,
        x[3] + u[1] * dt,
        x[4] + u[2] * dt,
    ]
end

function RD.discrete_dynamics!(model::DiscreteModelAuto, xn, x, u, t, dt)
    xn[1] = x[1] + x[3] * dt
    xn[2] = x[2] + x[4] * dt
    xn[3] = x[3] + u[1] * dt
    xn[4] = x[4] + u[2] * dt
    return nothing
end

function RD.jacobian!(model::DiscreteModelAuto, J, xn, x, u, t, dt)
    J .= 0
    J[1,1] = 1
    J[1,3] = dt
    J[2,2] = 1
    J[2,4] = dt
    J[3,3] = 1
    J[3,5] = dt
    J[4,4] = 1
    J[4,6] = dt
    return nothing
end

model = DiscreteModelAuto()
test_model(model)
test_error_allocs(model)