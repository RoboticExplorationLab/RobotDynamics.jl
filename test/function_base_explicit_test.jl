
struct TestFun0{CH} <: RobotDynamics.AbstractFunction
    cfg::ForwardDiff.JacobianConfig{Nothing, Float64, CH, Tuple{Vector{ForwardDiff.Dual{Nothing, Float64, CH}}, Vector{ForwardDiff.Dual{Nothing, Float64, CH}}}}
    cache::FiniteDiff.JacobianCache{Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}, UnitRange{Int64}, Nothing, Val{:forward}(), Float64}
    function TestFun0()
        n,m,p = 2,2,3
        cfg = ForwardDiff.JacobianConfig(nothing, zeros(p), zeros(n+m))
        cache = FiniteDiff.JacobianCache(zeros(n+m), zeros(p))
        new{length(cfg.seeds)}(cfg, cache)
    end
end
function RobotDynamics.evaluate(::TestFun0, x, u, p)
    return SA[cos(x[1]) * u[1], sin(x[2]^2 * x[1]) * u[2], exp(x[2] + x[1]/10)] * p[1]
end
function RobotDynamics.evaluate!(::TestFun0, y, x, u, p)
    y[1] = cos(x[1])  * u[1]
    y[2] = sin(x[2]^2 * x[1]) * u[2]
    y[3] = exp(x[2] + x[1]/10)
    y .*= p[1]
    return nothing
end
RobotDynamics.state_dim(::TestFun0) = 2
RobotDynamics.control_dim(::TestFun0) = 2
RobotDynamics.output_dim(::TestFun0) = 3

function RobotDynamics.jacobian!(::TestFun0, J, y, x, u, p)
    J .= 0
    J[1,1] = -sin(x[1]) * u[1]
    J[1,3] = cos(x[1])
    J[2,1] = x[2]^2 * cos(x[2]^2 * x[1]) * u[2]
    J[2,2] = 2 * x[1] * x[2] * cos(x[2]^2 * x[1]) * u[2]
    J[2,4] = sin(x[2]^2 * x[1])
    J[3,1] = exp(x[2] + x[1]/10) / 10
    J[3,2] = exp(x[2] + x[1]/10)
    J .*= p[1]
    return nothing
end

function RobotDynamics.jacobian!(::StaticReturn, ::ForwardAD, fun::TestFun0, J, y, z)
    f(_z) = RobotDynamics.evaluate(fun, getstate(z, _z), getcontrol(z, _z), getparams(z))
    J .= ForwardDiff.jacobian(f, getdata(z))
    return nothing
end

function RobotDynamics.jacobian!(::InPlace, ::ForwardAD, fun::TestFun0, J, y, z)
    f!(_y,_z) = RobotDynamics.evaluate!(fun, _y, getstate(z, _z), getcontrol(z, _z), getparams(z))
    ForwardDiff.jacobian!(J, f!, y, getdata(z), fun.cfg)
    return nothing
end

function RobotDynamics.jacobian!(::StaticReturn, ::FiniteDifference, fun::TestFun0, J, y, z)
    f!(_y,_z) = _y .= RobotDynamics.evaluate(fun, getstate(z, _z), getcontrol(z, _z), getparams(z))
    FiniteDiff.finite_difference_jacobian!(J, f!, getdata(z), fun.cache)
    return nothing
end

function RobotDynamics.jacobian!(::InPlace, ::FiniteDifference, fun::TestFun0, J, y, z)
    f!(_y,_z) = RobotDynamics.evaluate!(fun, _y, getstate(z, _z), getcontrol(z, _z), getparams(z))
    FiniteDiff.finite_difference_jacobian!(J, f!, getdata(z), fun.cache)
    return nothing
end

fun = TestFun0()
n,m,p = RD.dims(fun)
x = @SVector randn(n)
u = @SVector randn(m)
t = 1.2
dt = 0.1

zs = KnotPoint{n,m}([x;u],t,dt) 
z = KnotPoint{n,m}(Vector([x;u]),t,dt)
z_ = copy(z.z)
@test getstate(zs, z_) isa SVector{n}
@test getcontrol(zs, z_) isa SVector{m}
@test getstate(z, z_) isa SubArray 
@test getcontrol(z, z_) isa SubArray 

RD.evaluate(fun, zs)
y = zeros(3)
RD.evaluate!(fun, y, z)

test_fun(fun)