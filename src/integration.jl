"Fourth-order Runge-Kutta method with zero-order-old on the controls"
struct RK4 <: Explicit 
    k1::Vector{Float64}
    k2::Vector{Float64}
    k3::Vector{Float64}
    k4::Vector{Float64}
    function RK4(n::Integer, m::Integer)
        k1,k2,k3,k4 = zeros(n), zeros(n), zeros(n), zeros(n)
        new(k1,k2,k3,k4)
    end
end

function integrate(::RK4, model, x, u, t, h)
	k1 = dynamics(model, x,        u, t      )*h
	k2 = dynamics(model, x + k1/2, u, t + h/2)*h
	k3 = dynamics(model, x + k2/2, u, t + h/2)*h
	k4 = dynamics(model, x + k3,   u, t + h  )*h
	x + (k1 + 2k2 + 2k3 + k4)/6
end

function integrate!(int::RK4, model, xn, x, u, t, h)
    k1,k2,k3,k4 = int.k1, int.k2, int.k3, int.k4
    dynamics!(model, k1, x, u, t)
    @. xn = x + k1 * h/2
    dynamics!(model, k2, xn, u, t + h/2)
    @. xn = x + k2 * h/2
    dynamics!(model, k3, xn, u, t + h/2)
    @. xn = x + k3 * h
    dynamics!(model, k4, xn, u, t + h)
    @. xn = x + (k1 + 2k2 + 2k3 + k4)/6
    return nothing
end

function jacobian!(int::RK4, sig::StaticReturn, model, J, xn, z) 
    x,u,t,h = state(z), control(z), time(z), timestep(z)
    n,m = size(model)
    ix = SVector{n}(1:n)
    iu = SVector{m}(n+1:n+m)
	k1 = dynamics(model, x,        u, t      )*h
	k2 = dynamics(model, x + k1/2, u, t + h/2)*h
	k3 = dynamics(model, x + k2/2, u, t + h/2)*h
    ztmp = int.z

    set_state!(ztmp, x)
    set_control!(ztmp, u)
    set_time!(ztmp, t)
    set_timestep!(ztmp, h)
    jacobian!(sig, UserDefined(), model, J, xn, ztmp)
    A1,B1 = J[ix,ix], J[ix,iu] 

    set_state!(ztmp, x + k1/2)
    set_time!(ztmp, t + h/2)
    jacobian!(sig, UserDefined(), model, J, xn, ztmp)
    A2,B2 = J[ix,ix], J[ix,iu] 

    set_state!(ztmp, x + k2/2)
    set_time!(ztmp, t + h/2)
    jacobian!(sig, UserDefined(), model, J, xn, ztmp)
    A3,B3 = J[ix,ix], J[ix,iu] 

    set_state!(ztmp, x + k3)
    set_time!(ztmp, t + h)
    jacobian!(sig, UserDefined(), model, J, xn, ztmp)
    A4,B4 = J[ix,ix], J[ix,iu] 

    dA1 = A1 * h
    dA2 = A2 * (I + 0.5 * dA1) * h
    dA3 = A3 * (I + 0.5 * dA2) * h
    dA4 = A4 * (I + dA3) * h

    dB1 = B1 * h
    dB2 = B2 * h + 0.5 * A2 * dB1 * h
    dB3 = B3 * h + 0.5 * A3 * dB2 * h
    dB4 = B4 * h +       A4 * dB3 * h

    J[ix,ix] .= I + (dA1 + 2dA2 + 2dA3 + dA4) / 6
    J[ix,iu] .= (dB1 + 2db2 + 2dB3 + dB4) / 6
    return nothing
end

function jacobian!(int::RK4, sig::StaticReturn, model, J, xn, z) 
    x,u,t,h = state(z), control(z), time(z), timestep(z)
    k1,k2,k3,k4 = int.k1, int.k2, int.k3, int.k4
    A1,A2,A3,A4 = int.A[1], int.A[2], int.A[3], int.A[4]
    B1,B2,B3,B4 = int.B[1], int.B[2], int.B[3], int.B[4]
    dA1,dA2,dA3,dA4 = int.dA[1], int.dA[2], int.dA[3], int.dA[4]
    dB1,dB2,dB3,dB4 = int.dB[1], int.dB[2], int.dB[3], int.dB[4]
    ztmp = int.z
    n,m = size(model)
    ix,iu = 1:n, n+1:n+m

    dynamics!(model, k1, x, u, t)
    set_state!(ztmp, x)
    set_control!(ztmp, u)
    set_time!(ztmp, t)
    set_timestep!(ztmp, h)
    jacobian!(sig, UserDefined(), model, J, xn, ztmp)
    A1 .= @view J[ix,ix]
    B1 .= @view J[ix,iu]

    @. xn = x + k1 * h/2
    dynamics!(model, k2, xn, u, t + h/2)
    set_state!(ztmp, xn)
    set_time!(ztmp, t + h/2)
    jacobian!(sig, UserDefined(), model, J, xn, ztmp)
    A2 .= @view J[ix,ix]
    B2 .= @view J[ix,iu]

    @. xn = x + k2 * h/2
    dynamics!(model, k3, xn, u, t + h/2)
    set_state!(ztmp, xn)
    set_time!(ztmp, t + h/2)
    jacobian!(sig, UserDefined(), model, J, xn, ztmp)
    A3 .= @view J[ix,ix]
    B3 .= @view J[ix,iu]

    @. xn = x + k3 * h
    set_state!(ztmp, xn)
    set_time!(ztmp, t + h)
    jacobian!(sig, UserDefined(), model, J, xn, ztmp)
    A4 .= @view J[ix,ix]
    B4 .= @view J[ix,iu]

    # dA = A1 * h
    dA1 .= A1 .* h

    # dA2 = A2 * (I + 0.5 * dA1) * h
    dA2 .= I
    dA2 .+= 0.5 .* dA1
    mul!(dA2, A2, dA2)
    dA2 .*= h

    # dA3 = A3 * (I + 0.5 * dA2) * h
    dA3 .= I
    dA3 .+= 0.5 .* dA2
    mul!(dA3, A3, dA3)
    dA3 .*= h

    # dA4 = A4 * (I + dA3) * h
    dA4 .= I
    dA4 .+= dA3
    mul!(dA4, A4, dA4)
    dA3 .*= h

    # dB1 = B1 * h
    dB1 .= B1 .* h

    # dB2 = B2 * h + 0.5 * A2 * dB1 * h
    dB2 .= B2
    mul!(dB2, A2, dB1, 0.5 * h, h)

    # dB3 = B3 * h + 0.5 * A3 * dB2 * h
    dB3 .= B3
    mul!(dB3, A3, dB2, 0.5 * h, h)

    # dB4 = B4 * h + A4 * dB3 * h
    dB4 .= B4
    mul!(dB4, A4, dB3, h, h)

    J[ix,ix] .= I .+ (dA1 .+ 2dA2 .+ 2dA3 .+ dA4) ./ 6
    J[ix,iu] .= (dB1 .+ 2db2 .+ 2dB3 .+ dB4) ./ 6
    return nothing
end


############################################################################################
#                                  EXPLICIT METHODS 								       #
############################################################################################

function integrate(::Type{Euler}, model::AbstractModel, x::StaticVector, u::StaticVector, t, dt)
	xdot = dynamics(model, x, u, t)
	return x + xdot * dt
end

function integrate(::Type{RK2}, model::AbstractModel, x::StaticVector, u::StaticVector, t, dt)
	k1 = dynamics(model, x,        u, t       )*dt
	k2 = dynamics(model, x + k1/2, u, t + dt/2)*dt
	x + k2
end

function integrate(::Type{RK3}, model::AbstractModel, x::StaticVector, u::StaticVector,
		t, dt)
    k1 = dynamics(model, x,             u, t       )*dt;
    k2 = dynamics(model, x + k1/2,      u, t + dt/2)*dt;
    k3 = dynamics(model, x - k1 + 2*k2, u, t + dt  )*dt;
    x + (k1 + 4*k2 + k3)/6
end

function integrate(::Type{RK4}, model::AbstractModel, x::StaticVector, u::StaticVector, t, dt)
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
