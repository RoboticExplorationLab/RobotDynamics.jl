using RobotDynamics: orientation

# Set up model
model = Body{QuatRotation{Float64}}()
@test RD.dims(model) == (13, 6, 13)
x, u = rand(model)
t, dt = 0, 0.1
z = RD.KnotPoint(x, u, t, dt)
q = orientation(model, x)
ir, iq, iv, iω, iu = RobotDynamics.gen_inds(model)

# Test analytical continuous-time Jacobian
RobotDynamics.velocity_frame(::Body) = :world
function AD_jacobian!(
    ∇f::Matrix,
    model::RD.AbstractModel,
    z::RD.AbstractKnotPoint{n,m},
) where {n,m}
    ix, iu = SVector{n}(1:n), SVector{m}(n+1:n+m)
    t = z.t
    f_aug(z) = RD.dynamics(model, z[ix], z[iu], t)
    s = z.z
    ForwardDiff.jacobian!(∇f, f_aug, s)
end
F = zeros(13, 19)
AD_jacobian!(F, model, z)

F2 = zero(F)
y2 = zeros(13)
RD.jacobian!(RD.StaticReturn(), RD.ForwardAD(), model, F2, y2, z)
@test F2 ≈ F

ω = RD.angular_velocity(model, x)
ForwardDiff.jacobian(
    q -> Rotations.kinematics(QuatRotation(q, false), ω),
    Rotations.params(q),
)
Rotations.kinematics(q, ω)

D = RobotDynamics.DynamicsJacobian(13, 6)
RD.jacobian!(RD.StaticReturn(), RD.ForwardAD(), model, D, y2, z)
@test D ≈ F

# @btime AD_jacobian!($F, $model, $z)
# @btime jacobian!($F, $model, $z)

# Test in body frame
RobotDynamics.velocity_frame(::Body) = :body
F3 = zeros(13, 19)
AD_jacobian!(F3, model, z)
@test !(F3 ≈ F)
RD.jacobian!(RD.StaticReturn(), RD.ForwardAD(), model, D, y2, z)
@test D ≈ F3

dmodel = RD.DiscretizedDynamics{RD.RK4}(model)
RD.jacobian!(RD.StaticReturn(), RD.ForwardAD(), dmodel, F, y2, z)

RobotDynamics.velocity_frame(::Body) = :world
RD.jacobian!(RD.StaticReturn(), RD.ForwardAD(), dmodel, F2, y2, z)
@test !(F ≈ F2)
@test !(F ≈ D)
@test !(F2 ≈ D)

# Test state error functions
x0 = rand(model)[1]
dx = RobotDynamics.state_diff(model, x, x0)
@test dx[ir] ≈ (x-x0)[ir]
@test dx[iv.-1] ≈ (x-x0)[iv]
@test dx[iω.-1] ≈ (x-x0)[iω]
@test dx[4:6] ≈ orientation(model, x) ⊖ orientation(model, x0)

G0 = Rotations.∇differential(q)
G = zeros(RD.state_dim(model), RobotDynamics.errstate_dim(model))
@test size(G) == (13, 12)
RD.errstate_jacobian!(model, G, z)
@test G ≈ cat(I(3), G0, I(6), dims=(1, 2))

b = @SVector rand(13)
∇G0 = Rotations.∇²differential(q, b[SA[4, 5, 6, 7]])
∇G = zeros(RD.errstate_dim(model), RD.errstate_dim(model))
@test size(∇G) == (12, 12)
RobotDynamics.∇errstate_jacobian!(model, ∇G, x, b)
@test ∇G ≈ cat(zeros(3, 3), ∇G0, zeros(6, 6), dims=(1, 2))

# @btime RobotDynamics.state_diff_jacobian!($G, $model, $z)
# @btime RobotDynamics.∇²differential!($∇G, $model, $x, $b)