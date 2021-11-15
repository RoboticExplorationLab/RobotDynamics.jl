using StaticArrays
using RobotDynamics: KnotPoint, state,control
n,m = 10,5
x = @SVector rand(n)
u = @SVector rand(m)
t = 0.0
dt = 0.1
z = KnotPoint(x,u,t,dt)
RobotDynamics.setstate!(z, 2x)
RobotDynamics.setcontrol!(z, 2u)
@test RD.state(z) ≈ 2x
@test RD.control(z) ≈ 2u
RD.setdata!(z, 3*[x;u])
@test state(z) ≈ 3x
@test control(z) ≈ 3u
@test z isa KnotPoint

# Terminal time step
z = KnotPoint(x,u,t,dt)
@test RD.is_terminal(z) == false
@test RD.getdata(z) ≈ [x;u]
z.dt = 0.0
@test RD.is_terminal(z) == true

# StaticKnotPoint
t = 1.3
z = RD.StaticKnotPoint(x, u, t, dt)
@test_throws ErrorException RobotDynamics.setstate!(z, x)
@test_throws ErrorException RobotDynamics.setcontrol!(z, u)
@test_throws ErrorException RobotDynamics.setdata!(z, [x;u])

z0 = KnotPoint(2x,2u,t,dt)
z = RD.StaticKnotPoint(z0, [x;u])
@test state(z) ≈ x
@test control(z) ≈ u
@test z.dt ≈ dt
@test z.t ≈ 1.3
z = RD.StaticKnotPoint(z0)
@test state(z) ≈ 2x
@test control(z) ≈ 2u