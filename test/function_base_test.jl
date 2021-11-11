using RobotDynamics
using StaticArrays
using Test
using BenchmarkTools
using ForwardDiff
using FiniteDiff
using LinearAlgebra
using Random

using RobotDynamics: @autodiff, state, control, KnotPoint, getdata, getstate, getcontrol,
                     FiniteDifference, ForwardAD, StaticReturn, InPlace

using RobotDynamics: evaluate, evaluate!, jacobian!
function test_allocs(fun, params=nothing)
    n,m,p = size(fun)
    z_ = @SVector randn(n+m)
    z = KnotPoint{n,m}(z_,1.0,0.1)
    y = zeros(p)
    J = zeros(p,n+n)
    allocs = 0
    allocs += @allocated evaluate(fun, z, params)
    allocs += @allocated evaluate!(fun, y, z, params)
    allocs += @allocated jacobian!(RobotDynamics.InPlace(), RobotDynamics.ForwardAD(), fun, J, y, z, params)
    allocs += @allocated jacobian!(RobotDynamics.InPlace(), RobotDynamics.FiniteDifference(), fun, J, y, z, params)
    allocs += @allocated jacobian!(RobotDynamics.StaticReturn(), RobotDynamics.ForwardAD(), fun, J, y, z, params)
    allocs += @allocated jacobian!(RobotDynamics.StaticReturn(), RobotDynamics.FiniteDifference(), fun, J, y, z, params)

    # check for zero allocations with normal array inputs for in place methods
    z = KnotPoint{n,m}(Vector(z_),1.0,0.1)
    allocs += @allocated evaluate(fun, z, p)
    allocs += @allocated evaluate!(fun, y, z, p)
    allocs += @allocated jacobian!(RobotDynamics.InPlace(), RobotDynamics.ForwardAD(), fun, J, y, z, params)
    allocs += @allocated jacobian!(RobotDynamics.InPlace(), RobotDynamics.FiniteDifference(), fun, J, y, z, params)
    allocs += @allocated jacobian!(RobotDynamics.StaticReturn(), RobotDynamics.FiniteDifference(), fun, J, y, z, params)
end

function test_fun(fun, p=nothing)
    Random.seed!(1)
    @test size(fun) == (2,2,3)

    z_ = @SVector randn(4)
    z = KnotPoint{2,2}(z_,1.0,0.1)
    y = zeros(3)
    J = zeros(3,4)
    J0 = zeros(3,4)
    evaluate!(fun, y, z, p)
    @test y ≈ evaluate(fun, z, p)

    if isnothing(p)
        jacobian!(fun, J0, y, z)
    else
        jacobian!(fun, J0, y, z, p)
    end
    jacobian!(RobotDynamics.InPlace(), RobotDynamics.UserDefined(), fun, J, y, z, p)
    @test J ≈ J0
    jacobian!(RobotDynamics.StaticReturn(), RobotDynamics.UserDefined(), fun, J, y, z, p)
    @test J ≈ J0

    jacobian!(RobotDynamics.InPlace(), RobotDynamics.ForwardAD(), fun, J, y, z, p)
    @test J ≈ J0
    jacobian!(RobotDynamics.StaticReturn(), RobotDynamics.ForwardAD(), fun, J, y, z, p)
    @test J ≈ J0

    jacobian!(RobotDynamics.InPlace(), RobotDynamics.FiniteDifference(), fun, J, y, z, p)
    @test J ≈ J0 atol = 1e-6
    jacobian!(RobotDynamics.StaticReturn(), RobotDynamics.FiniteDifference(), fun, J, y, z, p)
    @test J ≈ J0 atol = 1e-6

    @test test_allocs(fun, p) == 0
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

function RobotDynamics.jacobian!(::StaticReturn, ::ForwardAD, fun::TestFun0, J, y, z, p)
    f(_z) = RobotDynamics.evaluate(fun, getstate(z, _z), getcontrol(z, _z), p)
    J .= ForwardDiff.jacobian(f, getdata(z))
    return nothing
end

function RobotDynamics.jacobian!(::InPlace, ::ForwardAD, fun::TestFun0, J, y, z, p)
    f!(_y,_z) = RobotDynamics.evaluate!(fun, _y, getstate(z, _z), getcontrol(z, _z), p)
    ForwardDiff.jacobian!(J, f!, y, getdata(z), fun.cfg)
    return nothing
end

function RobotDynamics.jacobian!(::StaticReturn, ::FiniteDifference, fun::TestFun0, J, y, z, p)
    f!(_y,_z) = _y .= RobotDynamics.evaluate(fun, getstate(z, _z), getcontrol(z, _z), p)
    FiniteDiff.finite_difference_jacobian!(J, f!, getdata(z), fun.cache)
    return nothing
end

function RobotDynamics.jacobian!(::InPlace, ::FiniteDifference, fun::TestFun0, J, y, z, p)
    f!(_y,_z) = RobotDynamics.evaluate!(fun, _y, getstate(z, _z), getcontrol(z, _z), p)
    FiniteDiff.finite_difference_jacobian!(J, f!, getdata(z), fun.cache)
    return nothing
end

fun = TestFun0()
n,m,p = size(fun)
x = @SVector randn(n)
u = @SVector randn(m)
t = 1.2
dt = 0.1
params = (t,dt)

zs = KnotPoint{n,m}([x;u],t,dt) 
z = KnotPoint{n,m}(Vector([x;u]),t,dt)
z_ = copy(z.z)
@test zs isa RobotDynamics.SKnotPoint
@test getstate(zs, z_) isa SVector{n}
@test getcontrol(zs, z_) isa SVector{m}
@test getstate(z, z_) isa SubArray 
@test getcontrol(z, z_) isa SubArray 

test_fun(fun, params)

##
@macroexpand @autodiff struct TestFun <: RobotDynamics.AbstractFunction 
    a::Int
    function TestFun()
        new(1.0)
    end
end ForwardAD StaticReturn

RobotDynamics.input_dim(::TestFun) = 4
RobotDynamics.output_dim(::TestFun) = 3

function RobotDynamics.evaluate(::TestFun, z)
    return SA[sin(z[1]), cos(z[2]), z[3] * exp(z[4])]
end

function RobotDynamics.evaluate!(::TestFun, y, z)
    y[1] = sin(z[1])
    y[2] = cos(z[2])
    y[3] = z[3] * exp(z[4])
end

function RobotDynamics.jacobian!(::TestFun, J, y, z)
    J .= 0
    J[1,1] = cos(z[1])
    J[2,2] = -sin(z[2])
    J[3,3] = exp(z[4])
    J[3,4] = z[3] * exp(z[4])
    return nothing
end

test_allocs(fun, params)


fun = TestFun()
@test fun.a == 1.0
test_fun(fun)

##############################
# With Input Parameters
##############################
using StaticArrays, ForwardDiff
@autodiff struct TestFunTime <: RobotDynamics.AbstractFunction end

function RobotDynamics.evaluate(fun::TestFunTime, z, p = (1,2))
    return SA[sin(z[1]) * p[1], cos(z[2]) * p[2], z[3] * exp(z[4])]
end
function RobotDynamics.evaluate!(::TestFunTime, y, z, p)
    y[1] = sin(z[1]) * p[1]
    y[2] = cos(z[2]) * p[2]
    y[3] = z[3] * exp(z[4])
end

function RobotDynamics.jacobian!(::TestFunTime, J, y, z, p)
    J .= 0
    J[1,1] = cos(z[1]) * p[1]
    J[2,2] = -sin(z[2]) * p[2]
    J[3,3] = exp(z[4])
    J[3,4] = z[3] * exp(z[4])
    return nothing
end
RobotDynamics.input_dim(::TestFunTime) = 4
RobotDynamics.output_dim(::TestFunTime) = 3

fun = TestFunTime()
z = @SVector zeros(4)
p = (3,2)
y = zeros(3)
J = zeros(3,4)
out = RobotDynamics.evaluate(fun, z, p)
RobotDynamics.evaluate!(fun, y, z, p)
@test y ≈ out

@test out[2] == p[2]

jacobian!(RobotDynamics.StaticReturn(), RobotDynamics.ForwardAD(), fun, J, y, z, p)
@test J[1,1] == p[1]
p = (4,2)
jacobian!(RobotDynamics.StaticReturn(), RobotDynamics.ForwardAD(), fun, J, y, z, p)
@test J[1,1] == p[1]

test_fun(fun, p)

##############################
# With Type Parameters
##############################
@autodiff struct TestFunParam{T} <: RobotDynamics.AbstractFunction 
    a::T
    function TestFunParam(a::T) where T
        new{T}(a)
    end
end

RobotDynamics.input_dim(::TestFunParam) = 4
RobotDynamics.output_dim(::TestFunParam) = 3

function RobotDynamics.evaluate(::TestFunParam, z)
    return SA[sin(z[1]), cos(z[2]), z[3] * exp(z[4])]
end

function RobotDynamics.evaluate!(::TestFunParam, y, z)
    y[1] = sin(z[1])
    y[2] = cos(z[2])
    y[3] = z[3] * exp(z[4])
end

function RobotDynamics.jacobian!(::TestFunParam, J, y, z)
    J .= 0
    J[1,1] = cos(z[1])
    J[2,2] = -sin(z[2])
    J[3,3] = exp(z[4])
    J[3,4] = z[3] * exp(z[4])
    return nothing
end

fun = TestFunParam(1.0)
@test fun.a == 1.0
test_fun(fun)

##############################
# Without inner constructor and inherited
##############################
abstract type TestFunBase <: RobotDynamics.AbstractFunction end

@macroexpand @autodiff struct TestFunInherited <: TestFunBase end
RobotDynamics.input_dim(::TestFunBase) = 4
RobotDynamics.output_dim(::TestFunBase) = 3

function RobotDynamics.evaluate(::TestFunBase, z)
    return SA[sin(z[1]), cos(z[2]), z[3] * exp(z[4])]
end

function RobotDynamics.evaluate!(::TestFunBase, y, z)
    y[1] = sin(z[1])
    y[2] = cos(z[2])
    y[3] = z[3] * exp(z[4])
end

function RobotDynamics.jacobian!(::TestFunBase, J, y, z)
    J .= 0
    J[1,1] = cos(z[1])
    J[2,2] = -sin(z[2])
    J[3,3] = exp(z[4])
    J[3,4] = z[3] * exp(z[4])
    return nothing
end

fun = TestFunInherited()
test_fun(fun)