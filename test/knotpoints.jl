using StaticArrays
n,m = 10,5
x = @SVector rand(n)
u = @SVector rand(m)
dt = 0.1
z = KnotPoint(x,u,dt)
RobotDynamics.set_state!(z, 2x)
RobotDynamics.set_control!(z, 2u)
@test state(z) ≈ 2x
@test control(z) ≈ 2u
RobotDynamics.set_z!(z, 3*[x;u])
@test state(z) ≈ 3x
@test control(z) ≈ 3u
@test z isa KnotPoint
@test z isa RobotDynamics.GeneralKnotPoint{T,N,M,SVector{NM,T}} where {T,N,M,NM}

# Terminal time step
z = KnotPoint(x,u,dt)
@test RobotDynamics.is_terminal(z) == false
@test RobotDynamics.get_z(z) ≈ [x;u]
z.dt = 0.0
@test RobotDynamics.is_terminal(z) == true
@test RobotDynamics.get_z(z) ≈ x

zterm = KnotPoint(x, m, 1.2)
@test control(zterm) ≈ zero(u)
@test state(zterm) ≈ x
@test RobotDynamics.is_terminal(zterm)

# GeneralKnotPoint
z_ = Vector([x;u])
z = RobotDynamics.GeneralKnotPoint(n,m,z_,dt)
@test z isa RobotDynamics.GeneralKnotPoint{Float64,n,m,Vector{Float64}}
@test RobotDynamics.get_z(z) isa Vector
@test state(z) isa SVector{n}
@test control(z) isa SVector{m}

# StaticKnotPoint
z = StaticKnotPoint(x, u, dt, 1.2)
@test_throws ErrorException RobotDynamics.set_state!(z, x)
@test_throws ErrorException RobotDynamics.set_control!(z, u)
@test_throws ErrorException RobotDynamics.set_z!(z, [x;u])

z0 = KnotPoint(2x,2u,dt,1.3)
z = StaticKnotPoint(z0, [x;u])
@test state(z) ≈ x
@test control(z) ≈ u
@test z.dt ≈ dt
@test z.t ≈ 1.3
z = StaticKnotPoint(z0)
@test state(z) ≈ 2x
@test control(z) ≈ 2u

z = z0 + z_
@test state(z) ≈ 3x
@test control(z) ≈ 3u
@test z isa StaticKnotPoint
@test z.dt ≈ z0.dt
@test z.t ≈ z0.t
z = z0 + 2*z0
@test state(z) ≈ 6x
@test control(z) ≈ 6u
@test z isa StaticKnotPoint
@test z.dt ≈ z0.dt
@test z.t ≈ z0.t

z *= 10
@test state(z) ≈ 60x
@test control(z) ≈ 60u
