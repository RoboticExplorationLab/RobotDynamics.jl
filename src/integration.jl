############################################################################################
#                                  IMPLICIT METHODS 								       #
############################################################################################

function discrete_dynamics(::Type{RK3}, model::AbstractModel, x::SVector{N,T}, u::SVector{M,T},
		t, dt) where {N,M,T}
    k1 = dynamics(model, x,             u, t       )*dt;
    k2 = dynamics(model, x + k1/2,      u, t + dt/2)*dt;
    k3 = dynamics(model, x - k1 + 2*k2, u, t       )*dt;
    x + (k1 + 4*k2 + k3)/6
end

function discrete_dynamics(::Type{RK2}, model::AbstractModel, x::SVector, u::SVector, t, dt)
	k1 = dynamics(model, x,        u, t       )*dt
	k2 = dynamics(model, x + k1/2, u, t + dt/2)*dt
	x + k2
end

function discrete_dynamics(::Type{RK4}, model::AbstractModel, x::SVector, u::SVector, t, dt)
	k1 = dynamics(model, x,        u, t       )*dt
	k2 = dynamics(model, x + k1/2, u, t + dt/2)*dt
	k3 = dynamics(model, x + k2/2, u, t + dt/2)*dt
	k4 = dynamics(model, x + k3,   u, t + dt  )*dt
	x + (k1 + 4k2 + 4k3 + k4)/6
end
