using RobotDynamics
using RobotDynamics: HybridModel
using StaticArrays

struct Unicycle <: AbstractModel end
RobotDynamics.state_dim(::Unicycle) = 3
RobotDynamics.control_dim(::Unicycle) = 2
RobotDynamics.dynamics(::Unicycle,x,u) = SA[u[1]*cos(x[3]), u[1]*sin(x[3]), u[2]]

# struct SLAM1{L} <: AbstractModel 
#     model::Unicycle
#     SLAM1(num_landmarks::Int) = new{2num_landmarks}()
# end
# RobotDynamics.state_dim(model::SLAM1) = state_dim(model.model)
# RobotDynamics.control_dim(model::SLAM1{L}) where L = control_dim(model.model) + L 
# RobotDynamics.next_state_dim(model::SLAM1{L}) where L = state_dim(model.model) + L 

# function discrete_dynamics(::Type{Q}, model::SLAM1{L}, x, u, t) where {Q,L}
#     x1 = SA[x[1], x[2], x[3]]
#     u1 = SA[u[1], u[2]]
#     x2 = discrete_dynamics(Q, model.model, x1, u1, t)
#     ℓ = pop(pop(u))  # landmark positions
#     return [x2; u2]
# end

struct SLAM{L} <: AbstractModel
    model::Unicycle
    SLAM(num_landmarks::Int) = new{2num_landmarks}()
end
RobotDynamics.state_dim(model::SLAM{L}) where L = state_dim(model.model) + L 
RobotDynamics.control_dim(model::SLAM) = control_dim(model.model)

function RobotDynamics.discrete_dynamics(::Type{Q}, model::SLAM, x, u, t, dt) where Q
    x1 = SA[x[1], x[2], x[3]]
    u1 = SA[u[1], u[2]]
    x2 = discrete_dynamics(Q, model.model, x1, u1, t, dt)
    ℓ = pop(pop(pop(x)))  # landmark positions
    return [x2; ℓ]
end


##
num_landmarks = 4
model0 = SLAM(num_landmarks)
state_dim(model0) == 3 + 2num_landmarks
control_dim(model0) == 2

slam_init(model::SLAM, x2, x0, u0, u_new) = [x2; u_new]
model = RobotDynamics.InitialControl(model0, 
    slam_init, SA[1,2], SVector{8}(1:8) .+ 2)

x = @SVector rand(3+8)
u0 = @SVector rand(2+8)
discrete_dynamics(RK4, model, x, u0, 0.0, 0.1)
@btime discrete_dynamics($RK4, $model, $x, $u0, 0.0, 0.1)

N = 11
tf = 5
times = range(0,tf, length=N)
model = HybridModel([model1, model2], [[1], 2:N], times)
model.model_inds == [[1]; fill(2,N-1)]
model.n == [[3]; fill(3 + 2num_landmarks, N-1)]
model.m == [[2 + 2num_landmarks]; fill(2, N-1)]

