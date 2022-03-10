using Test
using RobotDynamics
using StaticArrays
using ForwardDiff
using LinearAlgebra
using FiniteDiff
using Random
using Rotations
const RD = RobotDynamics

include("cartpole_model.jl")

##
function testcustomlu()
    n,m = 4,2
    J1 = randn(n, n+m)
    A = @view J1[:,1:n]
    A2 = copy(A)

    ipiv = similar(A, BLAS.BlasInt, min(n, n+m))
    # @allocated LAPACK.getrf!(A, ipiv)
    allocs = @allocated F = lu!(A, ipiv)
    F2 = lu!(A)
    @test F2.L ≈ F2.L
    @test F2.U ≈ F2.U
    @test allocs == 0
end
testcustomlu()

##
model = Cartpole()
integrator = RD.ImplicitMidpoint(model)
dmodel = RD.DiscretizedDynamics(model, integrator)
dmodel_rk4 = RD.DiscretizedDynamics(model, RD.RK4(model))
n,m = RD.dims(model)
x1 = zeros(n)
u1 = fill(1.0, m)
dt = 0.01
z1 = KnotPoint{n,m}(x1, u1, 0.0, dt)
x2_rk4 = RD.discrete_dynamics(dmodel_rk4, z1)

function midpoint_residual(x2, z1)
    n,m = RD.dims(z1)
    v = zeros(eltype(x2), n+m)
    z2 = RD.StaticKnotPoint(z1, v)
    RD.setstate!(z2, x2)
    RD.dynamics_error(dmodel, z2, z1)
end

function newton_solve(z1)
    n,m = RD.dims(z1)
    r(xn) = midpoint_residual(xn, z1)

    xn = copy(RD.state(z1))
    for i = 1:10
        res = r(xn)
        if norm(res) < 1e-6
            println("Converged in $i iterations.")
            break
        end

        ∇r = ForwardDiff.jacobian(r, xn)
        xn .-= ∇r\res
    end
    return xn
end
x2 = newton_solve(z1)
@test norm(midpoint_residual(x2, z1)) < 1e-6

# Test in-place evaluation
x2_inplace = copy(x1)
RD.evaluate!(dmodel, x2_inplace, z1)
@test norm(midpoint_residual(x2_inplace, z1)) < 1e-6

# Test static evaluation
z1s = KnotPoint(SVector{n}(x1), SVector{m}(u1), z1.t, z1.dt)
x2_static = RD.evaluate(dmodel, z1s)
@test norm(midpoint_residual(x2_static, z1s)) < 1e-6

# Test Jacobian
function fimplicit(v)
    z = RD.StaticKnotPoint(z1, v) 
    RD.evaluate(dmodel, z)
end
Jfd = FiniteDiff.finite_difference_jacobian(fimplicit, RD.getdata(z1))
J = zeros(n, n+m)
y = zeros(n)
ENV["JULIA_DEBUG"] = "RobotDynamics"
diff = RD.default_diffmethod(dmodel)
@test diff isa RD.ImplicitFunctionTheorem
@test diff === RD.ImplicitFunctionTheorem(RD.UserDefined())

z1.z .+= 1
@test_logs (:debug, r"Solving for next") RD.jacobian!(RD.InPlace(), diff, dmodel, J, y, z1)
@test_logs (:debug, r"Using cached") RD.jacobian!(RD.InPlace(), diff, dmodel, J, y, z1)
RD.setdata!(z1s, z1.z .+ 1)
@test_logs (:debug, r"Solving for next") RD.jacobian!(RD.InPlace(), diff, dmodel, J, y, z1s)
@test_logs (:debug, r"Using cached") RD.jacobian!(RD.StaticReturn(), diff, dmodel, J, y, z1s)

ENV["JULIA_DEBUG"] = ""
function test_allocs(dmodel, z::RD.AbstractKnotPoint{Nx,Nu}) where {Nx,Nu}
    zs = KnotPoint{Nx,Nu}(SVector{Nx+Nu}(RD.getdata(z)), z.t, z.dt)
    x2 = zeros(RD.state_dim(dmodel))
    allocs = @allocated RD.evaluate!(dmodel, x2, z1)
    allocs += @allocated RD.evaluate(dmodel, zs)
    J = zeros(Nx, Nx + Nu)
    y = zeros(Nx)
    diff = RD.default_diffmethod(dmodel)
    allocs += @allocated RD.jacobian!(RD.StaticReturn(), diff, dmodel, J, y, zs)
    allocs += @allocated RD.jacobian!(RD.InPlace(), diff, dmodel, J, y, z)
    return allocs
end
test_allocs(dmodel, z1)  # run once to compile non-logging versions
@test test_allocs(dmodel, z1) == 0
