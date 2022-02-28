using RobotDynamics: InPlace, StaticReturn, ForwardAD, FiniteDifference
using BenchmarkTools

struct MyFun{JCH} <: RobotDynamics.ScalarFunction 
    gradcfg::ForwardDiff.GradientConfig{Nothing, Float64, JCH, Vector{ForwardDiff.Dual{Nothing, Float64, JCH}}}
    hesscfg::ForwardDiff.HessianConfig{Nothing, Float64, JCH, Vector{ForwardDiff.Dual{Nothing, ForwardDiff.Dual{Nothing, Float64, JCH}, JCH}}, Vector{ForwardDiff.Dual{Nothing, Float64, JCH}}}
    gradcache::FiniteDiff.GradientCache{Nothing, Nothing, Nothing, Vector{Float64}, Val{:forward}(), Float64, Val{true}()}
    hesscache::FiniteDiff.HessianCache{Vector{Float64}, Val{:hcentral}(), Val{true}()}
    function MyFun()
        n,m = 4,2
        gradcfg = ForwardDiff.GradientConfig(nothing, zeros(n+m))
        hesscfg = ForwardDiff.HessianConfig(nothing, zeros(n+m))
        gradcache = FiniteDiff.GradientCache(zeros(n+m), zeros(n+m), Val(:forward))
        hesscache = FiniteDiff.HessianCache(zeros(n+m), zeros(n+m), zeros(n+m), zeros(n+m), Val(:hcentral), Val(true))
        new{length(gradcfg.seeds)}(gradcfg, hesscfg, gradcache, hesscache)
    end
end
RD.state_dim(::MyFun) = 4
RD.control_dim(::MyFun) = 2 

function RD.evaluate(::MyFun, x, u)
    x[1]^2 + cos(x[2] + 2u[2]) + u[1]*x[4] + x[3]^2
end

function RD.gradient!(::MyFun, grad, x, u)
    grad[1] = 2x[1]
    grad[2] = -sin(x[2] + 2u[2])
    grad[3] = 2x[3]
    grad[4] = u[1]
    grad[5] = x[4]
    grad[6] = -sin(x[2] + 2u[2]) * 2
    return nothing
end

function RD.hessian!(::MyFun, H, x, u)
    H .= 0
    H[1,1] = 2.0
    H[2,2] = -cos(x[2] + 2u[2])
    H[2,6] = -cos(x[2] + 2u[2]) * 2
    H[3,3] = 2
    H[4,5] = 1.0
    H[5,4] = 1.0
    H[6,2] = -cos(x[2] + 2u[2]) * 2
    H[6,6] = -cos(x[2] + 2u[2]) * 4
    return 
end

function RD.gradient!(::ForwardAD, fun::MyFun, grad, z)
    f(_z) = RD.evaluate(fun, RD.getstate(z, _z), RD.getcontrol(z, _z), RD.getparams(z))
    ForwardDiff.gradient!(grad, f, RD.getdata(z), fun.gradcfg)
    return nothing
end

function RD.hessian!(::ForwardAD, fun::MyFun, hess, z, cache=fun.hesscfg)
    f(_z) = RD.evaluate(fun, RD.getstate(z, _z), RD.getcontrol(z, _z), RD.getparams(z))
    ForwardDiff.hessian!(hess, f, RD.getdata(z), fun.hesscfg)
    return nothing
end

function hessian(::StaticReturn, ::ForwardAD, fun::MyFun, hess, z, cache=fun.hesscfg)
    f(_z) = RD.evaluate(fun, RD.getstate(z, _z), RD.getcontrol(z, _z), RD.getparams(z))
    hess .= ForwardDiff.hessian(f, RD.getdata(z), fun.hesscfg)
end

function RD.gradient!(::FiniteDifference, fun::MyFun, grad, z)
    f(_z) = RD.evaluate(fun, RD.getstate(z, _z), RD.getcontrol(z, _z), RD.getparams(z))
    FiniteDiff.finite_difference_gradient!(grad, f, RD.getdata(z), fun.gradcache)
    return nothing
end

function RD.hessian!(::FiniteDifference, fun::MyFun, hess, z, cache=fun.hesscache)
    cache.xmm .= z
    cache.xmp .= z
    cache.xpm .= z
    cache.xpp .= z
    f(_z) = RD.evaluate(fun, RD.getstate(z, _z), RD.getcontrol(z, _z), RD.getparams(z))
    FiniteDiff.finite_difference_hessian!(hess, f, RD.getdata(z), cache)
    return nothing
end

function test_scalar_fun(fun)
    n,m = RD.dims(fun)
    x = @SVector rand(n)
    u = @SVector rand(m)
    t,h = 1.0, 0.1
    grad = zeros(n+m)
    grad0 = zeros(n+m)
    hess = zeros(n+m, n+m)
    hess0 = zeros(n+m, n+m)
    zs = RD.KnotPoint(x, u, t, h)
    z = RD.KnotPoint{n,m}(Vector(x), Vector(u), t, h)

    RD.gradient!(fun, grad0, x, u)
    RD.gradient!(ForwardAD(), fun, grad, z)
    @test grad ≈ grad0
    RD.gradient!(FiniteDifference(), fun, grad, z)
    @test grad ≈ grad0 atol = 1e-6
    RD.gradient!(ForwardAD(), fun, grad, zs)
    @test grad ≈ grad0
    RD.gradient!(FiniteDifference(), fun, grad, zs)
    @test grad ≈ grad0 atol = 1e-6

    if run_alloc_tests
        allocs = 0
        allocs += @allocated RD.gradient!(ForwardAD(), fun, grad, z)
        @test allocs == 0
        allocs += @allocated RD.gradient!(ForwardAD(), fun, grad, zs)
        @test allocs == 0
        allocs += @allocated RD.gradient!(FiniteDifference(), fun, grad, z)
        @test allocs == 0
        allocs += @allocated RD.gradient!(FiniteDifference(), fun, grad, zs)
        @test allocs == 0
    end

    RD.hessian!(fun, hess0, x, u)
    RD.hessian!(ForwardAD(), fun, hess, z)
    @test hess ≈ hess0
    RD.hessian!(FiniteDifference(), fun, hess, z)
    @test hess ≈ hess0 atol = 1e-6
    RD.hessian!(ForwardAD(), fun, hess, zs)
    @test hess ≈ hess0
    RD.hessian!(FiniteDifference(), fun, hess, zs)
    @test hess ≈ hess0 atol = 1e-6

    if run_alloc_tests
        allocs += @allocated RD.hessian!(RD.UserDefined(), fun, hess, z)
        # allocs += @allocated RD.hessian!(ForwardAD(), fun, hess, z)  # this has some un-avoidable allocations
        allocs += @allocated RD.hessian!(ForwardAD(), fun, hess, zs)
        allocs += @allocated RD.hessian!(FiniteDifference(), fun, hess, z)
        allocs += @allocated RD.hessian!(FiniteDifference(), fun, hess, zs)
        @test allocs == 0
    end
end

##
@macroexpand RD.@autodiff struct MyFunAuto <: RobotDynamics.ScalarFunction end
RD.@autodiff struct MyFunAuto <: RobotDynamics.ScalarFunction end
RD.state_dim(::MyFunAuto) = 4
RD.control_dim(::MyFunAuto) = 2 

function RD.evaluate(::MyFunAuto, x, u)
    x[1]^2 + cos(x[2] + 2u[2]) + u[1]*x[4] + x[3]^2
end

function RD.gradient!(::MyFunAuto, grad, x, u)
    grad[1] = 2x[1]
    grad[2] = -sin(x[2] + 2u[2])
    grad[3] = 2x[3]
    grad[4] = u[1]
    grad[5] = x[4]
    grad[6] = -sin(x[2] + 2u[2]) * 2
    return nothing
end

function RD.hessian!(::MyFunAuto, H, x, u)
    H .= 0
    H[1,1] = 2.0
    H[2,2] = -cos(x[2] + 2u[2])
    H[2,6] = -cos(x[2] + 2u[2]) * 2
    H[3,3] = 2
    H[4,5] = 1.0
    H[5,4] = 1.0
    H[6,2] = -cos(x[2] + 2u[2]) * 2
    H[6,6] = -cos(x[2] + 2u[2]) * 4
    return 
end

##
@testset begin "Scalar Function"
    fun = MyFun()
    test_scalar_fun(fun)

    fun = MyFunAuto()
    test_scalar_fun(fun)
end