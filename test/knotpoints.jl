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
@test RD.vectype(z) === SVector{n+m,Float64}
zsim = similar(z)
@test zsim isa KnotPoint{n,m,MVector{n+m,Float64},Float64}

z2 = KnotPoint{n,m,Vector}(z.z, z.t, z.dt)
@test z2.z === z.z
@test RD.vectype(z2) === SVector{n+m,Float64}

# Terminal time step
z = KnotPoint(x,u,t,dt)
@test RD.is_terminal(z) == false
@test RD.getdata(z) ≈ [x;u]
z.dt = 0.0
@test RD.is_terminal(z) == true
@test RD.control(z) === @SVector zeros(m)

# Knotpoint with normal vector data
_z = Vector([x;u])
z = KnotPoint(n, m, _z, t, dt)
@test z isa KnotPoint{Any,Any}
@test _z === z.z
@test RD.state_dim(z) == n
@test RD.control_dim(z) == m

z = KnotPoint{n,m}(_z, t, dt)
@test z isa KnotPoint{n,m}
@test RD.state_dim(z) == n
@test RD.control_dim(z) == m
_z2 = 2*z
RD.setdata!(z, _z2)
@test _z ≈ _z2
@test RD.state(z) === view(_z,1:n)
@test RD.control(z) === view(_z,n+1:n+m)
RD.setcontrol!(z, u)
@test RD.control(z) ≈ u
z[2] = 10.1
@test RD.state(z)[2] === 10.1

@test RD.is_terminal(z) == false
z.dt = 0
@test RD.is_terminal(z) == true 
@test isempty(RD.control(z))
@test RD.vectype(z) === Vector{Float64}


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

z = RD.StaticKnotPoint(z0, [x;u])
z2 = RD.setdata(z, [2x;3u])
@test z2.z === [2x;3u]
z3 = RD.setstate(z, 3x)
@test z3.z === [3x;u]
z4 = RD.setcontrol(z, 4u)
@test z4.z === [x;4u]

z5 = 2*z
@test z5.z ===[2x; 2u]
@test z5.t == z.t
@test z5.dt == z.dt

z0 = RD.KnotPoint(x,u,t,dt)
z5 = 2*z0
@test z5.z ≈ [2x; 2u]
@test z5.t == z.t
@test z5.dt == z.dt

_z .= [x;u]
z_ = RD.KnotPoint{n,m}(_z,t,dt)
z5 = 2*z_
@test z5.z ≈ [2x; 2u]
@test z5.t == z.t
@test z5.dt == z.dt