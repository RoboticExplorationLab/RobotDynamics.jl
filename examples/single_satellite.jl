using RobotDynamics
using Rotations
using StaticArrays, LinearAlgebra
using BenchmarkTools

# Define the model struct to inherit from `RigidBody{R}`
struct Satellite{R,T} <: RigidBody{R}
    mass::T
    J::Diagonal{T,SVector{3,T}}
end
RobotDynamics.control_dim(::Satellite) = 6

# Define some simple "getter" methods that are required to evaluate the dynamics
RobotDynamics.mass(model::Satellite) = model.mass
RobotDynamics.inertia(model::Satellite) = model.J

# Define the 3D forces at the center of mass, in the world frame
function RobotDynamics.forces(model::Satellite, x::StaticVector, u::StaticVector)
    q = orientation(model, x)
    F = @SVector [u[1], u[2], u[3]]
    q*F  # world frame
end

# Define the 3D moments at the center of mass, in the body frame
function RobotDynamics.moments(model::Satellite, x::StaticVector, u::StaticVector)
    return @SVector [u[4], u[5], u[6]]  # body frame
end


# Build model
T = Float64
R = UnitQuaternion{T}
mass = 1.0
J = Diagonal(@SVector ones(3))
model = Satellite{R,T}(mass, J)

# Initialization
x,u = rand(model)
z = KnotPoint(x,u,0.1)
∇f = RobotDynamics.DynamicsJacobian(model)

# Continuous dynamics
dynamics(model, x, u)
jacobian!(∇f, model, z)

# Performance improvements for continuous dynamics Jacobian
b1 = @benchmark jacobian!($∇f, $model, $z)

function RobotDynamics.wrench_jacobian!(F, model::Satellite, z)
    x = state(z)
    u = control(z)
    q = orientation(model, x)
    ir, iq, iv, iω, iu = RobotDynamics.gen_inds(model)
    iF = SA[1,2,3]
    iM = SA[4,5,6]
    F[iF, iq] .= Rotations.∇rotate(q, u[iF])
    F[iF, iu[iF]] .= RotMatrix(q)
    for i = 1:3
        F[iM[i], iu[i+3]] = 1
    end
    return F
end
b2 = @benchmark jacobian!($∇f, $model, $z)

function RobotDynamics.wrench_sparsity(::Satellite)
    SA[false true  false false true;
       false false false false true]
end
b3 = @benchmark jacobian!($∇f, $model, $z)
println("Analytical Wrench Jacobian:  ", judge(minimum(b2), minimum(b1)))
println("w/ Wrench Jacobian Sparsity: ", judge(minimum(b3), minimum(b1)))

# Discrete dynamics
discrete_dynamics(RK2, model, z)
discrete_jacobian!(RK2, ∇f, model, z)
