using RobotDynamics
import Base: RefValue
const RD = RobotDynamics

using StaticArrays
using Test
using BenchmarkTools

n = 2
m = 1
A = @SMatrix [0.0 1.0; 0.0 0.0]
B = @SMatrix [0.0; 1.0]
@create_continuous_lti(DoubleIntegrator, n, m)
model = DoubleIntegrator()
set_A!(model, A)
set_B!(model, B)

@test RD.control_dim(model) == 1
@test RD.state_dim(model) == 2

@test RD.is_affine(model) == Val(false)
@test RD.is_time_varying(model) == false

@test model.A[] === A
@test model.B[] === B

@test RD.get_A(model, 10) === model.A[]
@test RD.get_B(model, 10) === model.B[]

dt = 0.01
t = 0.1

x, u = rand(model)

ẋ = dynamics(model, x, u, t)
@test A * x + B * u ≈ ẋ atol=1e-12


########################################################################################
A = [0.0 1.0;0.0 0.0]
B = [0.0;1.0]
d = [0.0;-9.81]

@create_continuous_lti(DoubleIntegratorAffine, n, m, true)
affine_model = DoubleIntegratorAffine()

set_A!(affine_model, A, 1)
set_B!(affine_model, B, 1)
set_d!(affine_model, d, 1)

@test affine_model.A[] == A
@test affine_model.B[] == SMatrix{2,1}(B)
@test affine_model.d[] == d

@test RD.get_A(affine_model, 10) === affine_model.A[]
@test RD.get_B(affine_model, 10) === affine_model.B[]
@test RD.get_d(affine_model, 10) === affine_model.d[]

ẋ = dynamics(affine_model, x, u, t)
@test affine_model.A[] * x + affine_model.B[] * u + affine_model.d[] ≈ ẋ atol=1e-12

###########################################################################################

@create_discrete_lti(DiscreteDoubleIntegrator, n, m, true)
discrete_model = DiscreteDoubleIntegrator()

dt = 0.01
discretize!(Exponential, discrete_model, affine_model, dt = dt)

cont_sys = zero(SizedMatrix{5, 5})
cont_sys[1:2, 1:2] .= get_A(affine_model)
cont_sys[1:2, 3:3] .= get_B(affine_model)
cont_sys[1:2, 4:5] .= oneunit(SizedMatrix{2, 2})

disc_sys = exp(cont_sys*dt)
A_d = disc_sys[1:2, 1:2]
B_d = disc_sys[1:2, 3:3]
d_d = disc_sys[1:2, 4:5]*get_d(affine_model)

@test A_d ≈ get_A(discrete_model)
@test B_d ≈ get_B(discrete_model)
@test d_d ≈ get_d(discrete_model)

@test discrete_dynamics(DiscreteSystemQuadrature, discrete_model, x, u, 0.01, dt) ≈ A_d*x + B_d*u + d_d

## Euler Test:
A_d = oneunit(SMatrix{2,2}) + get_A(affine_model) * dt
B_d = get_B(affine_model) * dt
d_d = get_d(affine_model) * dt
discretize!(Euler, discrete_model, affine_model, dt = dt)

@test A_d ≈ get_A(discrete_model)
@test B_d ≈ get_B(discrete_model)
@test d_d ≈ get_d(discrete_model)

