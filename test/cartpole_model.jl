
RD.@autodiff struct Cartpole{T} <: RobotDynamics.ContinuousDynamics
    mc::T
    mp::T
    l::T
    g::T
end

Cartpole() = Cartpole{Float64}(1.0, 0.2, 0.5, 9.81)

function RobotDynamics.dynamics(model::Cartpole, x, u)
    mc = model.mc  # mass of the cart in kg (10)
    mp = model.mp   # mass of the pole (point mass at the end) in kg
    l = model.l   # length of the pole in m
    g = model.g  # gravity m/s^2

    q = x[ @SVector [1,2] ]
    qd = x[ @SVector [3,4] ]

    s = sin(q[2])
    c = cos(q[2])

    H = @SMatrix [mc+mp mp*l*c; mp*l*c mp*l^2]
    C = @SMatrix [0 -mp*qd[2]*l*s; 0 0]
    G = @SVector [0, mp*g*l*s]
    B = @SVector [1, 0]

    qdd = -H\(C*qd + G - B*u[1])
    return [qd; qdd]
end

function RobotDynamics.dynamics!(model::Cartpole, xdot, x, u)
    mc = model.mc  # mass of the cart in kg (10)
    mp = model.mp   # mass of the pole (point mass at the end) in kg
    l = model.l   # length of the pole in m
    g = model.g  # gravity m/s^2

    q = x[ @SVector [1,2] ]
    qd = x[ @SVector [3,4] ]

    s = sin(q[2])
    c = cos(q[2])

    H = @SMatrix [mc+mp mp*l*c; mp*l*c mp*l^2]
    C = @SMatrix [0 -mp*qd[2]*l*s; 0 0]
    G = @SVector [0, mp*g*l*s]
    B = @SVector [1, 0]

    qdd = -H\(C*qd + G - B*u[1])
    xdot[1] = qd[1]
    xdot[2] = qd[2]
    xdot[3] = qdd[1]
    xdot[4] = qdd[2]
    return nothing
end

function RD.jacobian!(model::Cartpole, J, xdot, x, u, t)
    mc = model.mc  # mass of the cart in kg (10)
    mp = model.mp   # mass of the pole (point mass at the end) in kg
    l = model.l   # length of the pole in m
    g = model.g  # gravity m/s^2

    q = x[ @SVector [1,2] ]
    qd = x[ @SVector [3,4] ]

    s = sin(q[2])
    c = cos(q[2])

    H = @SMatrix [mc+mp mp*l*c; mp*l*c mp*l^2]
    C = @SMatrix [0 -mp*qd[2]*l*s; 0 0]
    G = @SVector [0, mp*g*l*s]
    B = @SVector [1, 0]
    qdd = -H\(C*qd + G - B*u[1])

    ∂H∂z = SA[
        0 -mp*l*s*qdd[2] 0 0 0
        0 -mp*l*s*qdd[1] 0 0 0
    ]
    ∂C∂z = SA[
        0 -mp*l*c*qd[2]^2 0 -2*mp*l*qd[2]*s 0
        0 0 0 0 0
    ]
    ∂G∂z = SA[
       0 0 0 0 0 
       0 mp*g*l*c 0 0 0
    ]
    ∂B∂z = SA[
        0 0 0 0 1
        0 0 0 0 0
    ]
    J[1:2,:] .= 0
    J[1,3] = 1.0
    J[2,4] = 1.0
    J[3:4,:] .= H\(-∂H∂z - ∂C∂z - ∂G∂z + ∂B∂z)
    return nothing
end

RobotDynamics.state_dim(::Cartpole) = 4
RobotDynamics.control_dim(::Cartpole) = 1