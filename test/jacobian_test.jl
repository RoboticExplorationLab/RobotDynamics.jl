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


############################################################################################
#                                    JACOBIAN-VECTOR PRODUCTS
############################################################################################
# model = Cartpole()
# λ = @SVector rand(state_dim(model))
# F = RD.DynamicsJacobian(model)
# x,u = rand(model)
# z = StaticKnotPoint(x,u,0.1)
# discrete_jacobian!(RK4, F, model, z)
# dg0 = F'λ
# jacobian!(F, model, z)
# g0 = F'λ

# RD.diffmethod(::Cartpole) = RD.ForwardAD()
# g = zeros(sum(size(model)))
# RD.discrete_jvp!(RK4, g, model, z, λ)
# @test g ≈ dg0
# @test (@allocated RD.discrete_jvp!(RK4, g, model, z, λ)) == 0

# RD.jvp!(g, model, z, λ)
# @test g ≈ g0
# @test (@allocated RD.jvp!(g, model, z, λ)) == 0

# # Finite difference
# RD.diffmethod(::Cartpole) = RD.FiniteDifference()
# cache = RD.gen_grad_cache(model)
# @test cache isa FiniteDiff.GradientCache
# g .= 0
# RD.discrete_jvp!(RK4, g, model, z, λ, cache)
# @test g ≈ dg0 atol=1e-6
# @test (@allocated RD.discrete_jvp!(RK4, g, model, z, λ, cache)) == 0

# RD.jvp!(g, model, z, λ, cache)
# @test g ≈ g0 atol=1e-6
# @test (@allocated RD.jvp!(g, model, z, λ, cache)) == 0

############################################################################################
#                               DYNAMICS HESSIANS
############################################################################################
# n,m = size(model)
# H = zeros(n+m,n+m)
# RD.∇jacobian!(H, model, z, λ)
# RD.∇discrete_jacobian!(RK4, H, model, z, λ)
