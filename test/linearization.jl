using RobotDynamics
using StaticArrays
using BenchmarkTools
const RD = RobotDynamics

nonlinear_model = Cartpole()

N = 5
dt = 0.01
n = state_dim(nonlinear_model)
m = control_dim(nonlinear_model)

Z = RD.Traj(n, m, dt, N)
times = RD.get_times(Z)

for knot_point in Z 
    knot_point.z = @SVector rand(n+m)
end


linearized_model = LinearizedModel(nonlinear_model, Z, dt=dt, integration=RK3, is_affine=true)

## Test that the linearization matches exactly at the knot points
for k=1:N-1
    z = Z[k]
    @test discrete_dynamics(RK3, nonlinear_model, z) ≈ 
        discrete_dynamics(PassThrough, linearized_model, z) atol=1e-4

    F1 = RD.DynamicsJacobian(nonlinear_model)
    F2 = RD.DynamicsJacobian(linearized_model)
    discrete_jacobian!(RK3, F1, nonlinear_model, z)
    discrete_jacobian!(PassThrough, F2, linearized_model, z)
    @test F1 ≈ F2
end

## Test with updated trajectory
for z in Z 
    z.z = @SVector rand(n+m)
end

update_trajectory!(linearized_model, Z)

for i=1:N-1
    z = Z[i]
    @test discrete_dynamics(RK3, nonlinear_model, z) ≈ 
        discrete_dynamics(PassThrough, linearized_model, z) atol=1e-4

    F1 = RD.DynamicsJacobian(nonlinear_model)
    F2 = RD.DynamicsJacobian(linearized_model)
    discrete_jacobian!(RK3, F1, nonlinear_model, z)
    discrete_jacobian!(PassThrough, F2, linearized_model, z)
    @test F1 ≈ F2
end

@test (@ballocated update_trajectory!($linearized_model, $Z) samples=2 evals=2) == 0
@test RD.integration(linearized_model) == RK3

## LinearModel
n,m = 10,5
A,B = gencontrollable(n,m)
d = rand(n)
model = LinearModel(A,B)
model_affine = LinearModel(A,B,d)

x,u = rand(model)
dynamics(model_affine, x, u)

# continuous -> continuous
linmodel = LinearizedModel(model)
@test RD.is_discrete(linmodel) == false
@test RD.get_A(linmodel) ≈ A 
@test RD.get_B(linmodel) ≈ B

# continuous -> discrete
dt = 0.01
linmodel = LinearizedModel(model, integration=RK2, dt=dt)
@test linmodel.Z[1].dt == dt
@test RD.is_discrete(linmodel)
@test RD.is_timevarying(linmodel) == false
@test RD.is_affine(linmodel) == false

@test RD.get_A(linmodel) ≈ I + A'*(I + 0.5*A*dt)*dt
@test RD.get_B(linmodel) ≈ (0.5*A*dt * B + B)*dt

# discrete -> continuous
model = LinearModel(A, B, dt=dt)
@test RD.is_discrete(model)
@test_throws AssertionError LinearizedModel(model)

# discrete -> discrete
linmodel = LinearizedModel(model, dt=dt, integration=PassThrough)
@test RD.get_A(linmodel) == A
@test RD.get_B(linmodel) == B

## Exponential
model = LinearModel(A, B)
@test RD.is_discrete(model) == false

linmodel = LinearizedModel(model, integration=Exponential, dt=dt)
RD.get_A(linmodel)
Ec = [A B; zeros(m,n+m)]
Ed = exp(Ec*dt)
@test Ed[1:n,1:n] ≈ RD.get_A(linmodel)
@test Ed[1:n,n .+ (1:m)] ≈ RD.get_B(linmodel)

## Try a large system
# dt = 0.01
# n,m = 150,50
# A,B = gencontrollable(n,m)
# e = n+m
# E = SizedMatrix{e,e}(zeros(e,e))
# @time Ed = RD.matrix_exponential!(E, A, B, dt)
# model = LinearModel(A,B)
# @test model.use_static == false
# linmodel = LinearizedModel(model, dt=dt, integration=Exponential)

# @btime RD.linearize!($linmodel)


###################################
## Exponential Discretization Tests
###################################

# using ControlSystems
# using StaticArrays

# dt = 0.01
# C_111 = ss([-5], [2], [3], [0])
# D_111 = c2d(C_111, dt, :zoh)
# exp([-5 2; 0 0]*dt)

# C_111_rd = LinearModel(-5*ones(1,1), 2*ones(1,1))
# D_111_rd = LinearizedModel(C_111_rd, dt=dt, integration=Exponential)

# @test D_111[1].A ≈ RD.get_A(D_111_rd)
# @test D_111[1].B ≈ RD.get_B(D_111_rd)


# ####
# C_212 = ss([-5 -3; 2 -9], [1; 2], [1 0; 0 1], [0; 0])
# D_212 = c2d(C_212, dt, :zoh)
# C_212_rd = LinearModel([-5 -3; 2 -9], reshape([1; 2], 2, 1))
# D_212_rd = LinearizedModel(C_212_rd, dt=dt, integration=Exponential)

# @test D_212[1].A ≈ RD.get_A(D_212_rd)
# @test D_212[1].B ≈ RD.get_B(D_212_rd)

# ###
# C_221 = ss([-5 -3; 2 -9], [1 0; 0 2], [1 0], [0 0])
# D_221 = c2d(C_221, dt, :zoh)
# C_221_rd = LinearModel([-5 -3; 2 -9], [1 0; 0 2])
# D_221_rd = LinearizedModel(C_221_rd, dt=dt, integration=Exponential)

# @test D_221[1].A ≈ RD.get_A(D_221_rd)
# @test D_221[1].B ≈ RD.get_B(D_221_rd)

# Constant Discrete Jacobian Tests:
n,m = 5, 5
A, B = gencontinuous(n,m)
random_model = LinearModel(A, B)
random_model_discrete = LinearModel(n, m; dt=dt)
x,u = rand(random_model)
z = KnotPoint(x, u, dt)

struct RandomLinear{n,m,T} <: AbstractModel
    A::SMatrix{n,n,T}
    B::SMatrix{n,m,T}
end
RD.state_dim(::RandomLinear{n}) = n
RD.control_dim(::RandomLinear{<:Any, m}) = m
RD.dynamics(model::RandomLinear, x, u) = model.A*x + model.B*u

random_model_test = RandomLinear{n,m,Float64}(A,B)
F = RD.DynamicsJacobian(n,m)

for RK in (RK2, RK3, RK4)
    linmodel = LinearizedModel(random_model_test, integration=RK, dt=dt)
    discrete_jacobian!(RK, F, random_model_test, z)
    @test RD.get_A(F) ≈ RD.get_A(linmodel) 
    @test RD.get_B(F) ≈ RD.get_B(linmodel) 
end
