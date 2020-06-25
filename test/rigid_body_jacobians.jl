# Set up model
model = Body{UnitQuaternion{Float64}}()
@test size(model) == (13,6)
x,u = rand(model)
z = KnotPoint(x,u,0.01)
q = orientation(model, x)
ir,iq,iv,iω,iu = RobotDynamics.gen_inds(model)

# Test analytical continuous-time Jacobian
RobotDynamics.velocity_frame(::Body) = :world
function AD_jacobian!(∇f::Matrix, model::AbstractModel, z::AbstractKnotPoint)
    ix, iu = z._x, z._u
	t = z.t
    f_aug(z) = dynamics(model, z[ix], z[iu], t)
    s = z.z
	ForwardDiff.jacobian!(∇f, f_aug, s)
end
F = zeros(13,19)
AD_jacobian!(F, model, z)

F2 = zero(F)
jacobian!(F2, model, z)
@test F2 ≈ F

ω = angular_velocity(model, x)
ForwardDiff.jacobian(q->Rotations.kinematics(UnitQuaternion(q,false), ω), Rotations.params(q))
Rotations.kinematics(q, ω)

D = RobotDynamics.DynamicsJacobian(13,6)
jacobian!(D, model, z)
@test D ≈ F

# @btime AD_jacobian!($F, $model, $z)
# @btime jacobian!($F, $model, $z)

# Test in body frame
RobotDynamics.velocity_frame(::Body) = :body
F3 = zeros(13,19)
AD_jacobian!(F3, model, z)
@test !(F3 ≈ F)
jacobian!(D, model, z)
@test D ≈ F3

# @btime AD_jacobian!($F, $model, $z)
# @btime jacobian!($F, $model, $z)


discrete_jacobian!(RK3, F2, model, z)
tmp = [RobotDynamics.DynamicsJacobian(13,6) for k = 1:3]
jacobian!(RK3, D, model, z, tmp)
@test F2 ≈ D

RobotDynamics.velocity_frame(::Body) = :world
discrete_jacobian!(RK3, F, model, z)
tmp = [RobotDynamics.DynamicsJacobian(13,6) for k = 1:3]
jacobian!(RK3, D, model, z, tmp)
@test F ≈ D

@test !(F ≈ F2)

# @btime discrete_jacobian!($RK3, $F, $model, $z)
# @btime jacobian!($RK3, $D, $model, $z, $tmp)


# Test state error functions
x0 = rand(model)[1]
dx = RobotDynamics.state_diff(model, x, x0)
@test dx[ir] == (x - x0)[ir]
@test dx[iv .- 1] == (x - x0)[iv]
@test dx[iω .- 1] == (x - x0)[iω]
@test dx[4:6] == orientation(model, x) ⊖ orientation(model, x0)

G0 = Rotations.∇differential(q)
G = zeros(state_dim(model), RobotDynamics.state_diff_size(model))
@test size(G) == (13,12)
RobotDynamics.state_diff_jacobian!(G, model, z)
@test G ≈ cat(I(3), G0, I(6), dims=(1,2))

b = @SVector rand(13)
∇G0 = Rotations.∇²differential(q, b[SA[4,5,6,7]])
∇G = zeros(RobotDynamics.state_diff_size(model), RobotDynamics.state_diff_size(model))
@test size(∇G) == (12,12)
RobotDynamics.∇²differential!(∇G, model, x, b)
@test ∇G ≈ cat(zeros(3,3), ∇G0, zeros(6,6), dims=(1,2))

# @btime RobotDynamics.state_diff_jacobian!($G, $model, $z)
# @btime RobotDynamics.∇²differential!($∇G, $model, $x, $b)
