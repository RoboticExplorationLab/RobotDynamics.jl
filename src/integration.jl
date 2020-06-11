############################################################################################
#                                  EXPLICIT METHODS 								       #
############################################################################################

function discrete_dynamics(::Type{RK3}, model::AbstractModel, x::StaticVector, u::StaticVector,
		t, dt)
    k1 = dynamics(model, x,             u, t       )*dt;
    k2 = dynamics(model, x + k1/2,      u, t + dt/2)*dt;
    k3 = dynamics(model, x - k1 + 2*k2, u, t + dt  )*dt;
    x + (k1 + 4*k2 + k3)/6
end

function discrete_dynamics(::Type{RK2}, model::AbstractModel, x::StaticVector, u::StaticVector, t, dt)
	k1 = dynamics(model, x,        u, t       )*dt
	k2 = dynamics(model, x + k1/2, u, t + dt/2)*dt
	x + k2
end

function discrete_dynamics(::Type{RK4}, model::AbstractModel, x::StaticVector, u::StaticVector, t, dt)
	k1 = dynamics(model, x,        u, t       )*dt
	k2 = dynamics(model, x + k1/2, u, t + dt/2)*dt
	k3 = dynamics(model, x + k2/2, u, t + dt/2)*dt
	k4 = dynamics(model, x + k3,   u, t + dt  )*dt
	x + (k1 + 2k2 + 2k3 + k4)/6
end

function jacobian!(::Type{RK3}, ∇f::DynamicsJacobian{n,nm}, model, z::AbstractKnotPoint, tmp) where {n,nm}
	m = nm-n
	# n,m = size(model)
	x = state(z)
	u = control(z)
	dt = z.dt
	t = z.t

	z1 = StaticKnotPoint([x; u], z._x, z._u, dt, t)
	k1 = dynamics(model, z1)*dt
	z2 = StaticKnotPoint([x + 0.5*k1; u], z._x, z._u, dt, t + 0.5*dt)
	k2 = dynamics(model, z2)*dt
	z3 = StaticKnotPoint([x - k1 + 2*k2; u], z._x, z._u, dt, t + dt)
	k3 = dynamics(model, z3)*dt
	jacobian!(tmp[1], model, z1)
	jacobian!(tmp[2], model, z2)
	jacobian!(tmp[3], model, z3)
	A1 = SMatrix{n,n}(tmp[1].A)
	A2 = SMatrix{n,n}(tmp[2].A)
	A3 = SMatrix{n,n}(tmp[3].A)
	B1 = SMatrix{n,m}(tmp[1].B)
	B2 = SMatrix{n,m}(tmp[2].B)
	B3 = SMatrix{n,m}(tmp[3].B)

	I3 = Diagonal(@SVector ones(n))
	∇f.A .= I3 + (A1*dt + 4*A2*(I3 + 0.5*A1*dt)*dt + A3*(I3 - A1*dt + 2*A2*(I3 + 0.5*A1*dt)*dt)*dt)/6
	∇f.B .= (B1*dt +
             4*(0.5*A2*B1*dt + B2)*dt +
             (A3*(-B1*dt + A2*B1*dt*dt + 2*B2*dt) + B3)*dt)/6
end
