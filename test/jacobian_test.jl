using Test
using StaticArrays
using RobotDynamics: DynamicsJacobian

n,m = 4,1
a = SizedMatrix{4,5}(rand(n,n+m))
F = DynamicsJacobian(a)
F[1,1] = 2
@test F[1,1] == 2
@test F[1,2] == a[1,2]
@test F.data === a

@test size(F.A) == (n,n)
@test size(F.B) == (n,m)
@test F.A == a[:,1:n]
@test F.B == a[:,n .+ (1:m)]

F.A[1,2] = 3
@test F[1,2] == 3
F.B[3,1] = 4
@test F[3,n+1] == 4

@test RobotDynamics.get_A(F) ≈ F.A
@test RobotDynamics.get_B(F) ≈ F.B
@test RobotDynamics.get_static_A(F) ≈ F.A
@test RobotDynamics.get_static_B(F) ≈ F.B
@test RobotDynamics.get_static_A(F) isa SMatrix{n,n}
@test RobotDynamics.get_static_B(F) isa SMatrix{n,m}

F_ = SMatrix(F)
@test DynamicsJacobian{n,n+m}(Tuple(F_)) isa DynamicsJacobian{n,n+m,Float64}
@test DynamicsJacobian{n,n+m,Float32}(Tuple(F_)) isa DynamicsJacobian{n,n+m,Float32}
@test Tuple(F) == Tuple(F_)
@test DynamicsJacobian(F_) isa DynamicsJacobian
@test DynamicsJacobian(F_) ≈ F

@test RobotDynamics.get_data(F_) === F_
@test RobotDynamics.get_data(F) === F.data.data
@test RobotDynamics.get_data(F.data) === F.data.data
@test RobotDynamics.get_data(F) isa Matrix
@test RobotDynamics.get_data(F.data) isa Matrix

## Test automatic diff methods
model = Cartpole()
@test RD.diffmethod(model) == RD.ForwardAD()
F1 = DynamicsJacobian(model)
F2 = copy(F1)
@test F2 isa DynamicsJacobian
x,u = rand(model)
z = KnotPoint(x,u,0.1)

jacobian!(F1, model, z)
RD._jacobian!(RD.FiniteDifference(), F2, model, z)
err = norm(F1-F2)
@test 1e-12 < err < 1e-6   # shouldn't be the same, but close

# pass in the cache
cache = FiniteDiff.JacobianCache(model, Val(:central), Float32)
@test eltype(cache.x1) == Float32
cache = FiniteDiff.JacobianCache(model, Val(:central))
RD._jacobian!(RD.FiniteDifference(), F2, model, z, cache)
@test eps(Float64) < norm(F1-F2) < err

# test discrete jacobian
discrete_jacobian!(RK4, F1, model, z)
@test (@allocated discrete_jacobian!(RK4, F1, model, z)) == 0
RD._discrete_jacobian!(RD.FiniteDifference(), RK4, F2, model, z)
err = norm(F1-F2)
@test eps(Float64) < err < 1e-6

RD.diffmethod(::Cartpole) = RD.FiniteDifference()
discrete_jacobian!(RK4, F2, model, z, cache)
@test eps(Float64) < norm(F1-F2) < err

# Only forward mode doesn't allocate
cache = FiniteDiff.JacobianCache(model)
discrete_jacobian!(RK4, F2, model, z, cache)
@test (@allocated discrete_jacobian!(RK4, F2, model, z, cache)) == 0  # it's 0 for b

RD.diffmethod(::Cartpole) = RD.ForwardAD()  # reset it