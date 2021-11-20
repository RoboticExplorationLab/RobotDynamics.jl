
using FiniteDiff
using Random
Random.seed!(1)
f(x) = x[1]^2 + cos(x[2] + 2x[6]) + x[5]*x[4] + x[3]^2
H = zeros(6,6)
x = randn(6)
cache = FiniteDiff.HessianCache(copy(x))
FiniteDiff.finite_difference_hessian!(H,f,x,cache)
H[1,1]

x = randn(6) * 1e6
FiniteDiff.finite_difference_hessian!(H,f,x,cache)
H[1,1]

cache.xmm .= x
cache.xmp .= x
cache.xpm .= x
cache.xpp .= x
FiniteDiff.finite_difference_hessian!(H,f,x,cache)
H[1,1]