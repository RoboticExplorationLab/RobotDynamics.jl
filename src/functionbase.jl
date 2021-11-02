import ForwardDiff, FiniteDiff
using StaticArrays, BenchmarkTools

abstract type AbstractFunction end

abstract type FunctionSignature end
struct InPlace <: FunctionSignature end
struct StaticReturn <: FunctionSignature end
function_signature(fun::AbstractFunction) = InPlace()

abstract type DiffMethod end
struct ForwardAD <: DiffMethod end
struct FiniteDifference <: DiffMethod end
struct UserDefined <: DiffMethod end
diff_method(fun::AbstractFunction) = UserDefined

state_dim(fun::AbstractFunction) = state_dim(typeof(fun))
control_dim(fun::AbstractFunction) = control_dim(typeof(fun))


## 
abstract type DI{D} <: AbstractFunction where D end
state_dim(::Type{<:DI{D}}) where D = 2D
control_dim(::Type{<:DI{D}}) where D = D

function call_jacobian(::DifMethod, ::FunctionSignature, fun::AbstractFunction, J, y, z)
    
end

@generated function evaluate(fun::DI{D}, x, u) where {D}
    N, M = 2D, D
    vel = [:(x[$i]) for i = M+1:N]
    us = [:(u[$i]) for i = 1:M]
    :(SVector{$N}($(vel...),$(us...)))
end
evaluate(model, x, u)

@generated function evaluate!(fun::DI{D}, y, x, u) where D
    N, M = 2D, D
    vel = [:(y[$(i - M)] = x[$i]) for i = M+1:N]
    us = [:(y[$(i + M)] = u[$i]) for i = 1:M]
    quote
        $(Expr(:block, vel...)) 
        $(Expr(:block, us...)) 
        return nothing
    end
end


@generated function jacobian!(fun::DI{D}, J, y, x, u) where D
    N, M = 2D, D
    jac = [:(J[$i, $(i+M)] = 1.0) for i = 1:N]
    quote
        $(Expr(:block, jac...))
        return nothing
    end
end



## Example
mutable struct DoubleIntegrator{D, NM} <: DI{D} 
    cfg::ForwardDiff.JacobianConfig{Nothing, Float64, NM, Tuple{Vector{ForwardDiff.Dual{Nothing, Float64, NM}}, Vector{ForwardDiff.Dual{Nothing, Float64, NM}}}}
    cache::FiniteDiff.JacobianCache{Vector{Float64}, Vector{Float64}, Vector{Float64}, UnitRange{Int64}, Nothing, Val{:forward}(), Float64}
    function DoubleIntegrator{D}() where D
        n = 2D
        m = D
        y = zeros(n)
        z = zeros(n+m)
        cfg = ForwardDiff.JacobianConfig(nothing, y, z)
        cache = FiniteDiff.JacobianCache(z, y)
        new{D, length(cfg.seeds)}(cfg, cache)
    end
end

function jacobian_ad!(fun::DoubleIntegrator{D}, J, y, z) where D
    f_aug!(y, z) = evaluate!(fun, y, view(z, 1:2D), view(z, 2D+1:3D))
    ForwardDiff.jacobian!(J, f_aug!, y, z, fun.cfg)
end

function jacobian_ad!(fun::DoubleIntegrator{D}, J, z) where D
    ix = SVector{2D}(1:2D)
    iu = SVector{D}(2D +1:3D)
    f_aug(z) = evaluate(fun, z[ix], z[iu]) 
    J .= ForwardDiff.jacobian(f_aug, z)
end

function jacobian_fd!(fun::DoubleIntegrator{D}, J, y, z; cache=fun.cache) where D
    f_aug!(y, z) = evaluate!(fun, y, view(z, 1:2D), view(z, 2D+1:3D))
    FiniteDiff.finite_difference_jacobian!(J, f_aug!, z, cache)
end

function jacobian_fd!(fun::DoubleIntegrator{D}, J, z) where D
    f_aug!(y, z) = y .= evaluate(fun, view(z, 1:2D), view(z, 2D+1:3D))
    FiniteDiff.finite_difference_jacobian!(J, f_aug!, z, fun.cache)
end

##
@autodiff mutable struct DoubleIntegrator2{D} <: DI{D}
    function DoubleIntegrator2{D}() where D
        n = 2D
        m = D
        new{D}()
    end
end FiniteDifference InPlace ForwardAD StaticReturn

##
dim = 30 
model = DoubleIntegrator{dim}()
model = DoubleIntegrator2{dim}()


n,m = 2dim, dim
y = zeros(n)
x = @SVector randn(n)
u = @SVector randn(m)
z = [x;u]
J = zeros(n, n+m)

@time y2 = evaluate(model, x, u)
@time evaluate!(model, y, x, u)
y2 ≈ y

@time jacobian!(model, J, y, x, u)  # 0.07 sec  w/ dim = 30
@time jacobian_ad!(model, J, y, z)  # 1.5 sec   w/ dim = 30
@time jacobian_ad!(model, J, z)     # 45 sec    w/ dim = 30

@time jacobian_fd!(model, J, y, z)  # 0.25 sec  w/ dim = 30
@time jacobian_fd!(model, J, z)     # 0.25 sec  w/ dim = 30

@btime evaluate($model, $x, $u)          # 5 ns
@btime evaluate!($model, $y, $x, $u)     # 8 ns

@btime jacobian!($model, $J, $y, $x, $u) # 14 ns
@btime jacobian_ad!($model, $J, $y, $z)  # 9.8 us
@btime jacobian_ad!($model, $J, $z)      # 330 ns
@btime jacobian_fd!($model, $J, $y, $z)  # 4.7 us
@btime jacobian_fd!($model, $J, $z)      # 6.1 us



##

function pass_kwargs(; kwargs...)
    @show typeof(kwargs)

    return kwargs
    test_fun(; kwargs...)
end

function test_fun(; a = 1.0, b = 2.1, apple=:apple)

end

kwargs = pass_kwargs(apple=true, a=2, b = "true")
kwargs_vec = collect(kwargs)
kwargs2 = Base.Iterators.Pairs((true, 2), (apple=Bool, a=1))
typeof(kwargs2)
pass_kwargs(apple=true)
test_fun(;kwargs2...)