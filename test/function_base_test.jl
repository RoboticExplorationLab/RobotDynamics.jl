using RobotDynamics
using StaticArrays
using Test
using BenchmarkTools
using ForwardDiff
using FiniteDiff
using LinearAlgebra
using Random
const RD = RobotDynamics

using RobotDynamics: @autodiff, state, control, KnotPoint, getdata, getstate, getcontrol, getparams,
                     FiniteDifference, ForwardAD, StaticReturn, InPlace

using RobotDynamics: evaluate, evaluate!, jacobian!
function test_allocs(fun)
    n,m,p = RD.dims(fun)
    z_ = @SVector randn(n+m)
    z = KnotPoint{n,m}(z_,1.0,0.1)
    test_allocs(fun, z)
end
function test_allocs(fun, z)
    n = RD.state_dim(z)
    m = RD.control_dim(z)
    z_ = RD.getdata(z)
    p = RD.output_dim(fun)
    w = RD.input_dim(fun)
    y = zeros(p)
    J = zeros(p,w)
    allocs = 0
    allocs += @allocated evaluate(fun, z)
    allocs += @allocated evaluate!(fun, y, z)
    allocs += @allocated jacobian!(RobotDynamics.InPlace(), RobotDynamics.ForwardAD(), fun, J, y, z)
    allocs += @allocated jacobian!(RobotDynamics.InPlace(), RobotDynamics.FiniteDifference(), fun, J, y, z)
    allocs += @allocated jacobian!(RobotDynamics.StaticReturn(), RobotDynamics.ForwardAD(), fun, J, y, z)
    allocs += @allocated jacobian!(RobotDynamics.StaticReturn(), RobotDynamics.FiniteDifference(), fun, J, y, z)

    # check for zero allocations with normal array inputs for in place methods
    z = KnotPoint{n,m}(Vector(z_),1.0,0.1)
    allocs += @allocated evaluate(fun, z)
    allocs += @allocated evaluate!(fun, y, z)
    allocs += @allocated jacobian!(RobotDynamics.InPlace(), RobotDynamics.ForwardAD(), fun, J, y, z)
    allocs += @allocated jacobian!(RobotDynamics.InPlace(), RobotDynamics.FiniteDifference(), fun, J, y, z)
    allocs += @allocated jacobian!(RobotDynamics.StaticReturn(), RobotDynamics.FiniteDifference(), fun, J, y, z)
end

function test_fun(fun)
    Random.seed!(1)
    @test RD.dims(fun) == (2,2,3)

    z_ = @SVector randn(4)
    z = KnotPoint{2,2}(z_,1.0,0.1)
    y = zeros(3)
    J = zeros(3,4)
    J0 = zeros(3,4)

    evaluate!(fun, y, z)
    @test y ≈ evaluate(fun, z)

    jacobian!(fun, J0, y, z)

    jacobian!(RobotDynamics.InPlace(), RobotDynamics.UserDefined(), fun, J, y, z)
    @test J ≈ J0
    jacobian!(RobotDynamics.StaticReturn(), RobotDynamics.UserDefined(), fun, J, y, z)
    @test J ≈ J0

    jacobian!(RobotDynamics.InPlace(), RobotDynamics.ForwardAD(), fun, J, y, z)
    @test J ≈ J0
    jacobian!(RobotDynamics.StaticReturn(), RobotDynamics.ForwardAD(), fun, J, y, z)
    @test J ≈ J0

    jacobian!(RobotDynamics.InPlace(), RobotDynamics.FiniteDifference(), fun, J, y, z)
    @test J ≈ J0 atol = 1e-6
    jacobian!(RobotDynamics.StaticReturn(), RobotDynamics.FiniteDifference(), fun, J, y, z)
    @test J ≈ J0 atol = 1e-6

    run_alloc_tests && (@test test_allocs(fun) == 0)
end


## 
struct TestFun0{CH} <: RobotDynamics.AbstractFunction
    cfg::ForwardDiff.JacobianConfig{Nothing, Float64, CH, Tuple{Vector{ForwardDiff.Dual{Nothing, Float64, CH}}, Vector{ForwardDiff.Dual{Nothing, Float64, CH}}}}
    cache::FiniteDiff.JacobianCache{Vector{Float64}, Vector{Float64}, Vector{Float64}, UnitRange{Int64}, Nothing, Val{:forward}(), Float64}
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

test_fun(fun)

##############################
# Autogen With Inner Constructor 
##############################
@autodiff struct TestFun <: RobotDynamics.AbstractFunction 
    a::Int
    function TestFun()
        new(1.0)
    end
end

RobotDynamics.state_dim(::TestFun) = 2
RobotDynamics.control_dim(::TestFun) = 2
RobotDynamics.output_dim(::TestFun) = 3

function RobotDynamics.evaluate(::TestFun, x, u)
    return SA[sin(x[1]), cos(x[2]), u[1] * exp(u[2])]
end

function RobotDynamics.evaluate!(::TestFun, y, x, u)
    y[1] = sin(x[1])
    y[2] = cos(x[2])
    y[3] = u[1] * exp(u[2])
end

function RobotDynamics.jacobian!(::TestFun, J, y, x, u)
    J .= 0
    J[1,1] = cos(x[1])
    J[2,2] = -sin(x[2])
    J[3,3] = exp(u[2])
    J[3,4] = u[1] * exp(u[2])
    return nothing
end

fun = TestFun()
@test :cfg ∈ fieldnames(typeof(fun))
@test :cache ∈ fieldnames(typeof(fun))
@test fun.a == 1.0
test_fun(fun)

##############################
# With Input Parameters
##############################
@autodiff struct TestFunTime <: RobotDynamics.AbstractFunction end

function RobotDynamics.evaluate(::TestFunTime, x, u, t)
    return SA[sin(x[1]) * t.t, cos(x[2]) * t.t, u[1] * exp(u[2])]
end

function RobotDynamics.evaluate!(::TestFunTime, y, x, u, t)
    y[1] = sin(x[1]) * t.t
    y[2] = cos(x[2]) * t.t
    y[3] = u[1] * exp(u[2])
end

function RobotDynamics.jacobian!(::TestFunTime, J, y, x, u, t)
    J .= 0
    J[1,1] = cos(x[1]) * t.t
    J[2,2] = -sin(x[2]) * t.t
    J[3,3] = exp(u[2])
    J[3,4] = u[1] * exp(u[2])
    return nothing
end
RobotDynamics.state_dim(::TestFunTime) = 2
RobotDynamics.control_dim(::TestFunTime) = 2
RobotDynamics.output_dim(::TestFunTime) = 3

fun = TestFunTime()
z_ = @SVector zeros(4)
z = KnotPoint{2,2}(z_, 1.2, 0.1)
y = zeros(3)
J = zeros(3,4)
out = RobotDynamics.evaluate(fun, z)
RobotDynamics.evaluate!(fun, y, z)
@test y ≈ out
@test out[2] == z.t 

jacobian!(RobotDynamics.StaticReturn(), RobotDynamics.ForwardAD(), fun, J, y, z)
@test J[1,1] == z.t 
z.t = 2.4
jacobian!(RobotDynamics.StaticReturn(), RobotDynamics.ForwardAD(), fun, J, y, z)
@test J[1,1] == z.t 

test_fun(fun)

##############################
# With Type Parameters
##############################
@autodiff struct TestFunParam{T} <: RobotDynamics.AbstractFunction 
    a::T
    function TestFunParam(a::T) where T
        new{T}(a)
    end
end

RobotDynamics.state_dim(::TestFunParam) = 2
RobotDynamics.control_dim(::TestFunParam) = 2
RobotDynamics.output_dim(::TestFunParam) = 3

function RobotDynamics.evaluate(::TestFunParam, x, u)
    return SA[sin(x[1]), cos(x[2]), u[1] * exp(u[2])]
end

function RobotDynamics.evaluate!(::TestFunParam, y, x, u)
    y[1] = sin(x[1])
    y[2] = cos(x[2])
    y[3] = u[1] * exp(u[2])
end

function RobotDynamics.jacobian!(::TestFunParam, J, y, x, u)
    J .= 0
    J[1,1] = cos(x[1])
    J[2,2] = -sin(x[2])
    J[3,3] = exp(u[2])
    J[3,4] = u[1] * exp(u[2])
    return nothing
end

fun = TestFunParam(1.0)
@test fun.a == 1.0
test_fun(fun)

##############################
# Without inner constructor and inherited
##############################
abstract type TestFunBase <: RobotDynamics.AbstractFunction end

@autodiff struct TestFunInherited <: TestFunBase end
RobotDynamics.state_dim(::TestFunBase) = 2
RobotDynamics.control_dim(::TestFunBase) = 2
RobotDynamics.output_dim(::TestFunBase) = 3

function RobotDynamics.evaluate(::TestFunBase, x, u)
    return SA[sin(x[1]), cos(x[2]), u[1] * exp(u[2])]
end

function RobotDynamics.evaluate!(::TestFunBase, y, x, u)
    y[1] = sin(x[1])
    y[2] = cos(x[2])
    y[3] = u[1] * exp(u[2])
end

function RobotDynamics.jacobian!(::TestFunBase, J, y, x, u)
    J .= 0
    J[1,1] = cos(x[1])
    J[2,2] = -sin(x[2])
    J[3,3] = exp(u[2])
    J[3,4] = u[1] * exp(u[2])
    return nothing
end

fun = TestFunInherited()
test_fun(fun)

##############################
# One-liner inner constructor
##############################
@autodiff struct TestInner <: TestFunBase 
    a::Int
    TestInner(a) = new(a)
end

fun = TestInner(1)
test_fun(fun)


##############################
## State-Only 
##############################
@autodiff struct TestFunStateOnly <: RobotDynamics.AbstractFunction end

RD.state_dim(::TestFunStateOnly) = 3
RD.output_dim(::TestFunStateOnly) = 2
RD.functioninputs(::TestFunStateOnly) = RD.StateOnly()

function RD.evaluate(::TestFunStateOnly, x::RD.DataVector)
    return SA[cos(x[1] - x[3]), sin(x[2]^2 * x[1])]
end

function RD.evaluate!(::TestFunStateOnly, y, x::RD.DataVector)
    y[1] = cos(x[1] - x[3])
    y[2] = sin(x[2]^2 * x[1])
    return nothing
end

function RD.jacobian!(::TestFunStateOnly, J, y, x::RD.DataVector)
    J .= 0
    J[1,1] = -sin(x[1] - x[3])
    J[1,3] = sin(x[1] - x[3])
    J[2,1] = cos(x[2]^2 * x[1]) * x[2]^2
    J[2,2] = cos(x[2]^2 * x[1]) * 2 * x[2] * x[1]
    return nothing
end

fun = TestFunStateOnly()
@test RD.functioninputs(fun) == RD.StateOnly()

n = RD.state_dim(fun)
p = RD.output_dim(fun)
x = @SVector randn(n)
u = @SVector randn(p)
y = zeros(p)
J = zeros(p,n)
J0 = copy(J)
fun = TestFunStateOnly()
zs = RD.KnotPoint(x,u,0.0,0.1)
z = RD.KnotPoint(Vector(x),Vector(u),0.0,0.1)

evaluate!(fun, y, z)
@test y ≈ evaluate(fun, z) ≈ evaluate(fun, x)
evaluate!(fun, y, x)
@test y ≈ evaluate(fun, z) ≈ evaluate(fun, x)

jacobian!(fun, J0, y, x)
jacobian!(RD.InPlace(), RD.UserDefined(), fun, J, y, z)
@test J ≈ J0
jacobian!(RD.StaticReturn(), RD.UserDefined(), fun, J, y, z)
@test J ≈ J0

jacobian!(RD.StaticReturn(), RD.ForwardAD(), fun, J, y, z)
@test J ≈ J0
jacobian!(RD.InPlace(), RD.ForwardAD(), fun, J, y, z)
@test J ≈ J0
jacobian!(RD.StaticReturn(), RD.FiniteDifference(), fun, J, y, z)
@test J ≈ J0 atol=1e-5
jacobian!(RD.InPlace(), RD.FiniteDifference(), fun, J, y, z)
@test J ≈ J0 atol=1e-5

@test RD.getinput(RD.functioninputs(fun), z) === state(z)
run_alloc_tests && @test test_allocs(fun, zs) == 0

##############################
## Control Only
##############################
@autodiff struct TestFunControlOnly <: RobotDynamics.AbstractFunction end

RD.control_dim(::TestFunControlOnly) = 2
RD.output_dim(::TestFunControlOnly) = 3
RD.functioninputs(::TestFunControlOnly) = RD.ControlOnly()

function RD.evaluate(::TestFunControlOnly, u::RD.DataVector)
    SA[exp(10*u[1]/u[2]), 1/u[1], (u[1] - u[2])^2]
end

function RD.evaluate!(::TestFunControlOnly, y, u::RD.DataVector)
    y[1] = exp(10*u[1]/u[2])
    y[2] = 1/u[1]
    y[3] = (u[1] - u[2])^2
    return nothing
end

function RD.jacobian!(::TestFunControlOnly, J, y, u::RD.DataVector)
    J .= 0
    e = exp(10*u[1] / u[2])
    J[1,1] = 10/u[2] * e
    J[1,2] = -10*u[1]/(u[2]^2) * e
    J[2,1] = -1/u[1]^2
    J[3,1] = 2*(u[1] - u[2])
    J[3,2] = -2*(u[1] - u[2])
    return nothing
end

fun = TestFunControlOnly()

m = RD.control_dim(fun)
p = RD.output_dim(fun)
x = @SVector randn(4)
u = @SVector randn(m)
y = zeros(p)
J = zeros(p,m)
J0 = copy(J)
zs = RD.KnotPoint(x,u,0.0,0.1)
z = RD.KnotPoint(Vector(x),Vector(u),0.0,0.1)

evaluate!(fun, y, z)
@test y ≈ evaluate(fun, z) ≈ evaluate(fun, u)
evaluate!(fun, y, u)
@test y ≈ evaluate(fun, z) ≈ evaluate(fun, u)

jacobian!(fun, J0, y, u)
jacobian!(RD.InPlace(), RD.UserDefined(), fun, J, y, z)
@test J ≈ J0
J
jacobian!(RD.StaticReturn(), RD.UserDefined(), fun, J, y, z)
@test J ≈ J0

jacobian!(RD.StaticReturn(), RD.ForwardAD(), fun, J, y, z)
@test J ≈ J0
jacobian!(RD.InPlace(), RD.ForwardAD(), fun, J, y, z)
@test J ≈ J0
jacobian!(RD.StaticReturn(), RD.FiniteDifference(), fun, J, y, z)
@test J ≈ J0 atol=1e-2
jacobian!(RD.InPlace(), RD.FiniteDifference(), fun, J, y, z)
@test J ≈ J0 atol=1e-2

@test RD.getinput(RD.functioninputs(fun), z) === control(z)
run_alloc_tests && @test test_allocs(fun, zs) == 0