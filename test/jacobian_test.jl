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

F_ = SMatrix(F)
@test DynamicsJacobian{n,n+m}(Tuple(F_)) isa DynamicsJacobian{n,n+m,Float64}
@test DynamicsJacobian{n,n+m,Float32}(Tuple(F_)) isa DynamicsJacobian{n,n+m,Float32}
@test Tuple(F) == Tuple(F_)

@test RobotDynamics.get_data(F_) === F_
@test RobotDynamics.get_data(F) === F.data.data
@test RobotDynamics.get_data(F.data) === F.data.data
@test RobotDynamics.get_data(F) isa Matrix
@test RobotDynamics.get_data(F.data) isa Matrix
