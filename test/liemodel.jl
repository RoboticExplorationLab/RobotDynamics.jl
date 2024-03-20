struct Satellite <: RD.LieGroupModel
    J::Diagonal{Float64,SVector{3,Float64}}
end

Satellite() = Satellite(Diagonal(@SVector ones(3)))

RobotDynamics.control_dim(::Satellite) = 3
Base.position(::Satellite, x::SVector) = @SVector zeros(3)
RobotDynamics.orientation(::Satellite, x::SVector) = QuatRotation(x[4], x[5], x[6], x[7])

RobotDynamics.LieState(::Satellite) = RobotDynamics.LieState(QuatRotation{Float64}, (3, 0))

function RobotDynamics.dynamics(model::Satellite, x::SVector, u::SVector)
    ω = @SVector [x[1], x[2], x[3]]
    q = normalize(@SVector [x[4], x[5], x[6], x[7]])
    J = model.J

    ωdot = J \ (u - ω × (J * ω))
    qdot = 0.5 * lmult(q) * hmat() * ω
    return [ωdot; qdot]
end

model = Satellite()
@test LieState(model) === RobotDynamics.QuatState(7, (4,))
@test RD.state_dim(model) == 7
@test RD.errstate_dim(model) == 6

x, u = rand(model)
s = LieState(model)
@test all(RobotDynamics.vec_states(s, x) .≈ (x[1:3], Float64[]))
@test all(RobotDynamics.rot_states(s, x) .≈ (QuatRotation(x[4:7]),))
@test all(RobotDynamics.vec_states(model, x) .≈ (x[1:3], Float64[]))
@test all(RobotDynamics.rot_states(model, x) .≈ (QuatRotation(x[4:7]),))

x2 = rand(s)
@test norm(x[4:7]) ≈ 1

dx = RobotDynamics.state_diff(model, x, x2)
@test length(dx) == 6
@test dx ≈ [x[1:3] - x2[1:3]; QuatRotation(x[4:7]) ⊖ QuatRotation(x2[4:7])]
