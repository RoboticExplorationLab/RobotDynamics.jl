"""
    ADVector{T}

A cache of vectors for making it easier to ForwardDiff. A vector of the desired type 
    is extracted by indexing:

    vec = adv[eltype(x)]

Since ForwardDiff Duals are parameterized on the chunk size, this type will store any 
number of vectors for different chunk sizes. A new vector of duals will be automatically
created (once) if it doesn't exist yet.

For best performance (by avoiding type-instability), annotate the indexing operation:

    vec = adv[T]::Vector{T}

While the annotation could have been included in the indexing operation itself, some 
benchmark results found it significantly faster to apply the annotation at the call site.

# Constructor

    ADVector{T}(n)

where `n` is the size of the vectors, and `T` is the numeric type (typically `Float64`).
"""
struct ADVector{T}
    v::Vector{T}
    d::Dict{DataType,Vector{D} where {D<:ForwardDiff.Dual{Nothing,T}}}
    function ADVector{T}(n::Integer) where {T}
        new{T}(zeros(T, n), Dict{DataType,Vector{D} where D}())
    end

    # Copy constructor
    # Does a deep copy of the cache dictionary
    function ADVector(adv::ADVector{T}) where T
        new{T}(copy(adv.v), deepcopy(adv.d))
    end
end
Base.getindex(adv::ADVector, ::Type{<:Number}) = adv.v
function Base.getindex(adv::ADVector, T::Type{<:ForwardDiff.Dual})
    if haskey(adv.d, T)
        return adv.d[T]
    else
        d = zeros(T, length(adv.v))
        adv.d[T] = d
        return d
    end
end
Base.copy(adv::ADVector{T}) where T = ADVector(adv)  # calls copy constructor

########################################
# Explicit Methods
########################################

"""
    Euler

Explicit Euler integration:

```math
x_{k+1} = x_k + h f(x_k, u_k)
```
where ``h`` is the time step.

!!! warning
    In general, explicit Euler integration **SHOULD NOT BE USED!** It is the worst possible 
    integration method since it is very inaccurate and can easily go unstable.
"""
struct Euler <: Explicit
    Euler(n::Integer, m::Integer) = new()
end

function integrate(::Euler, model, x, u, t, h)
    xdot = dynamics(model, x, u, t)
    return x + h * xdot
end

function integrate!(int::Euler, model, xn, x, u, t, h)
    dynamics!(model, xn, x, u, t)
    xn .*= h
    xn .+= x
    return nothing
end

function jacobian!(int::Euler, sig::FunctionSignature, model::ContinuousDynamics, J, xn, x, u, t, h)
    # Call the user-defined Continuous-time Jacobian
    jacobian!(model, J, xn, x, u, t)
    J .*= h
    for i = 1:state_dim(model)
        J[i, i] += 1.0
    end
    return nothing
end

@doc raw"""
    RK3

A third-order explicit Runge-Kutta method:

```math
\begin{aligned}
&k_1 = f(x_k, u_k, t) h \\
&k_2 = f(x_k + \frac{1}{2} k_1, u_k, t, + \frac{1}{2} h) h \\
&k_3 = f(x_k - k_1 + 2 k_2, u_k, t, + h) h \\
&x_{k+1} = x_k + \frac{1}{6} (k_1 + 4 k_2 + k_3) 
\end{aligned}
```
"""
struct RK3 <: Explicit 
    k1::ADVector{Float64}
    k2::ADVector{Float64}
    k3::ADVector{Float64}
    A::Vector{Matrix{Float64}}
    B::Vector{Matrix{Float64}}
    dA::Vector{Matrix{Float64}}
    dB::Vector{Matrix{Float64}}
end
function RK3(n::Integer, m::Integer)
    k1, k2 = ADVector{Float64}(n), ADVector{Float64}(n)
    k3 = ADVector{Float64}(n)
    A = [zeros(n, n) for i = 1:3]
    B = [zeros(n, m) for i = 1:3]
    dA = [zeros(n, n) for i = 1:3]
    dB = [zeros(n, m) for i = 1:3]
    RK3(k1, k2, k3, A, B, dA, dB)
end
getks(int::RK3, ::Type{T}) where {T} =
    int.k1[T]::Vector{T}, int.k2[T]::Vector{T}, int.k3[T]::Vector{T}

function integrate(::RK3, model, x, u, t, h)
    k1 = dynamics(model, x,            u, t      ) * h
    k2 = dynamics(model, x + k1 / 2,   u, t + h/2) * h
    k3 = dynamics(model, x - k1 + 2k2, u, t + h  ) * h
    return x + (k1 + 4k2 + k3) / 6
end

function integrate!(int::RK3, model, xn, x, u, t, h)
    T = eltype(xn)
    k1, k2, k3 = getks(int, T)
    dynamics!(model, k1, x, u, t)
    @. xn = x + k1 * h / 2
    dynamics!(model, k2, xn, u, t + h / 2)
    @. xn = x - k1 * h + 2 * k2 * h
    dynamics!(model, k3, xn, u, t + h)
    @. xn = x + h * (k1 + 4k2 + k3) / 6
    return nothing
end

function jacobian!(int::RK3, sig::StaticReturn, model, J, xn, x, u, t, h)
    n, m = dims(model)
    ix = SVector{n}(1:n)
    iu = SVector{m}(n+1:n+m)
    k1 = dynamics(model, x,            u, t      ) * h
    k2 = dynamics(model, x + k1 / 2,   u, t + h/2) * h

    jacobian!(model, J, xn, x, u, t)
    A1, B1 = J[ix, ix], J[ix, iu]

    jacobian!(model, J, xn, x + k1 / 2, u, t + h / 2)
    A2, B2 = J[ix, ix], J[ix, iu]

    jacobian!(model, J, xn, x - k1 + 2k2, u, t + h)
    A3, B3 = J[ix, ix], J[ix, iu]

    dA1 = A1 * h
    dA2 = A2 * (I + 0.5 * dA1) * h
    dA3 = A3 * (I - dA1 + 2 * dA2) * h

    dB1 = B1 * h
    dB2 = B2 * h + 0.5 * A2 * dB1 * h
    dB3 = B3 * h + A3 * (2dB2 - dB1) * h

    J[ix, ix] .= I + (dA1 + 4dA2 + dA3) / 6
    J[ix, iu] .= (dB1 + 4dB2 + dB3) / 6

    return nothing
end

function jacobian!(int::RK3, sig::InPlace, model, J, xn, x, u, t, h)
    # x,u,t,h = state(z), control(z), time(z), timestep(z)
    k1, k2, k3 = getks(int, Float64)
    A1, A2, A3 = int.A[1], int.A[2], int.A[3]
    B1, B2, B3 = int.B[1], int.B[2], int.B[3]
    dA1, dA2, dA3 = int.dA[1], int.dA[2], int.dA[3]
    dB1, dB2, dB3 = int.dB[1], int.dB[2], int.dB[3]
    n, m = dims(model)
    ix, iu = 1:n, n+1:n+m

    jacobian!(model, J, k1, x, u, t)
    dynamics!(model,    k1, x, u, t)
    A1 .= @view J[ix, ix]
    B1 .= @view J[ix, iu]

    @. xn = x + k1 * h / 2
    jacobian!(model, J, k2, xn, u, t + h / 2)
    dynamics!(model,    k2, xn, u, t + h / 2)
    A2 .= @view J[ix, ix]
    B2 .= @view J[ix, iu]

    @. xn = x - k1 * h + 2 * k2 * h
    jacobian!(model, J, k3, xn, u, t + h)
    dynamics!(model,    k3, xn, u, t + h)
    A3 .= @view J[ix, ix]
    B3 .= @view J[ix, iu]

    # dA = A1 * h
    dA1 .= A1 .* h

    # dA2 = A2 * (I + 0.5 * dA1) * h
    mul!(dA2, A2, dA1, 0.5, 0.0)
    dA2 .+= A2
    dA2 .*= h

    # dA3 = A3 * (I - dA1 + 2 * dA2) * h
    mul!(dA3, A3, dA2, 2.0, 0.0)
    mul!(dA3, A3, dA1,-1.0, 1.0)
    dA3 .+= A3
    dA3 .*= h

    # dB1 = B1 * h
    dB1 .= B1 .* h

    # dB2 = B2 * h + 0.5 * A2 * dB1 * h
    dB2 .= B2
    mul!(dB2, A2, dB1, 0.5, 1.0)
    dB2 .*= h

    # dB3 = B3 * h + A3 * (2dB2 - dB1) * h
    dB3 .= B3
    mul!(dB3, A3, dB2, 2.0, 1.0)
    mul!(dB3, A3, dB1,-1.0, 1.0)
    dB3 .*= h

    @. J[ix, ix] = (dA1 + 4dA2 + dA3) / 6
    for i = 1:n
        J[i, i] += 1.0
    end
    @. J[ix, iu] = (dB1 + 4dB2 + dB3) / 6

    return nothing
end

@doc raw"""
    RK4

The classic fourth-order explicit Runge-Kutta method.

```math
\begin{aligned}
& k_1 = f(x_k, u_k, t) h \\
& k_2 = f(x_k + \frac{1}{2} k_1, u_k, t + \frac{1}{2} h) h \\
& k_3 = f(x_k + \frac{1}{2} k_2, u_k, t + \frac{1}{2} h) h \\
& k_4 = f(x_k +  k_3, u_k, t + h) h \\
& x_{k+1} = x_k + \frac{1}{6}(k_1 + 2k_2 + 2k_3 + k_4)
\end{aligned}
```
"""
struct RK4 <: Explicit
    k1::ADVector{Float64}
    k2::ADVector{Float64}
    k3::ADVector{Float64}
    k4::ADVector{Float64}
    A::Vector{Matrix{Float64}}
    B::Vector{Matrix{Float64}}
    dA::Vector{Matrix{Float64}}
    dB::Vector{Matrix{Float64}}
end
function RK4(n::Integer, m::Integer)
    k1, k2 = ADVector{Float64}(n), ADVector{Float64}(n)
    k3, k4 = ADVector{Float64}(n), ADVector{Float64}(n)
    A = [zeros(n, n) for i = 1:4]
    B = [zeros(n, m) for i = 1:4]
    dA = [zeros(n, n) for i = 1:4]
    dB = [zeros(n, m) for i = 1:4]
    RK4(k1, k2, k3, k4, A, B, dA, dB)
end
getks(int::RK4, ::Type{T}) where {T} =
    int.k1[T]::Vector{T}, int.k2[T]::Vector{T}, int.k3[T]::Vector{T}, int.k4[T]::Vector{T}

function integrate(::RK4, model, x, u, t, h)
    k1 = dynamics(model, x, u, t) * h
    k2 = dynamics(model, x + k1 / 2, u, t + h / 2) * h
    k3 = dynamics(model, x + k2 / 2, u, t + h / 2) * h
    k4 = dynamics(model, x + k3, u, t + h) * h
    x + (k1 + 2k2 + 2k3 + k4) / 6
end

function integrate!(int::RK4, model, xn, x, u, t, h)
    T = eltype(xn)
    k1, k2, k3, k4 = getks(int, T)
    dynamics!(model, k1, x, u, t)
    @. xn = x + k1 * h / 2
    dynamics!(model, k2, xn, u, t + h / 2)
    @. xn = x + k2 * h / 2
    dynamics!(model, k3, xn, u, t + h / 2)
    @. xn = x + k3 * h
    dynamics!(model, k4, xn, u, t + h)
    @. xn = x + h * (k1 + 2k2 + 2k3 + k4) / 6
    return nothing
end

function jacobian!(int::RK4, sig::StaticReturn, model, J, xn, x, u, t, h)
    n, m = dims(model)
    ix = SVector{n}(1:n)
    iu = SVector{m}(n+1:n+m)
    k1 = dynamics(model, x, u, t) * h
    k2 = dynamics(model, x + k1 / 2, u, t + h / 2) * h
    k3 = dynamics(model, x + k2 / 2, u, t + h / 2) * h

    jacobian!(model, J, xn, x, u, t)
    A1, B1 = J[ix, ix], J[ix, iu]

    jacobian!(model, J, xn, x + k1 / 2, u, t + h / 2)
    A2, B2 = J[ix, ix], J[ix, iu]

    jacobian!(model, J, xn, x + k2 / 2, u, t + h / 2)
    A3, B3 = J[ix, ix], J[ix, iu]

    jacobian!(model, J, xn, x + k3, u, t + h)
    A4, B4 = J[ix, ix], J[ix, iu]

    dA1 = A1 * h
    dA2 = A2 * (I + 0.5 * dA1) * h
    dA3 = A3 * (I + 0.5 * dA2) * h
    dA4 = A4 * (I + dA3) * h

    dB1 = B1 * h
    dB2 = B2 * h + 0.5 * A2 * dB1 * h
    dB3 = B3 * h + 0.5 * A3 * dB2 * h
    dB4 = B4 * h + A4 * dB3 * h


    J[ix, ix] .= I + (dA1 + 2dA2 + 2dA3 + dA4) / 6
    J[ix, iu] .= (dB1 + 2dB2 + 2dB3 + dB4) / 6

    return nothing
end

function jacobian!(int::RK4, sig::InPlace, model, J, xn, x, u, t, h)
    # x,u,t,h = state(z), control(z), time(z), timestep(z)
    k1, k2, k3, k4 = getks(int, Float64)
    A1, A2, A3, A4 = int.A[1], int.A[2], int.A[3], int.A[4]
    B1, B2, B3, B4 = int.B[1], int.B[2], int.B[3], int.B[4]
    dA1, dA2, dA3, dA4 = int.dA[1], int.dA[2], int.dA[3], int.dA[4]
    dB1, dB2, dB3, dB4 = int.dB[1], int.dB[2], int.dB[3], int.dB[4]
    n, m = dims(model)
    ix, iu = 1:n, n+1:n+m

    jacobian!(model, J, k1, x, u, t)
    dynamics!(model,    k1, x, u, t)
    A1 .= @view J[ix, ix]
    B1 .= @view J[ix, iu]

    @. xn = x + k1 * h / 2
    jacobian!(model, J, k2, xn, u, t + h / 2)
    dynamics!(model,    k2, xn, u, t + h / 2)
    A2 .= @view J[ix, ix]
    B2 .= @view J[ix, iu]

    @. xn = x + k2 * h / 2
    jacobian!(model, J, k3, xn, u, t + h / 2)
    dynamics!(model,    k3, xn, u, t + h / 2)
    A3 .= @view J[ix, ix]
    B3 .= @view J[ix, iu]

    @. xn = x + k3 * h
    jacobian!(model, J, k4, xn, u, t + h)
    A4 .= @view J[ix, ix]
    B4 .= @view J[ix, iu]


    # dA = A1 * h
    dA1 .= A1 .* h

    # dA2 = A2 * (I + 0.5 * dA1) * h
    mul!(dA2, A2, dA1, 0.5, 0.0)
    dA2 .+= A2
    dA2 .*= h

    # dA3 = A3 * (I + 0.5 * dA2) * h
    mul!(dA3, A3, dA2, 0.5, 0.0)
    dA3 .+= A3
    dA3 .*= h

    # dA4 = A4 * (I + dA3) * h
    mul!(dA4, A4, dA3, 1.0, 0.0)
    dA4 .+= A4
    dA4 .*= h

    # dB1 = B1 * h
    dB1 .= B1 .* h

    # dB2 = B2 * h + 0.5 * A2 * dB1 * h
    dB2 .= B2
    mul!(dB2, A2, dB1, 0.5, 1.0)
    dB2 .*= h

    # dB3 = B3 * h + 0.5 * A3 * dB2 * h
    dB3 .= B3
    mul!(dB3, A3, dB2, 0.5, 1.0)
    dB3 .*= h

    # dB4 = B4 * h + A4 * dB3 * h
    mul!(dB4, A4, dB3)
    dB4 .+= B4
    dB4 .*= h

    @. J[ix, ix] = (dA1 + 2dA2 + 2dA3 + dA4) / 6
    for i = 1:n
        J[i, i] += 1.0
    end
    @. J[ix, iu] = (dB1 + 2dB2 + 2dB3 + dB4) / 6

    return nothing
end

########################################
# Implicit Methods
########################################

# Use Newton's method to solve for the next state for implicit dynamics
function integrate(integrator::Implicit, model::ImplicitDynamicsModel, 
                   z::AbstractKnotPoint{Nx,Nu}) where {Nx,Nu} 
    integrate(integrator, model, state(z), control(z), time(z), timestep(z))
end
function integrate(integrator::Implicit, model::ImplicitDynamicsModel, 
                   x::AbstractVector, u::AbstractVector, t, h)
    Nx = length(x)
    Nu = length(u)
    integrate(integrator, model, SizedVector{Nx}(x), SizedVector{Nu}(u), t, h)
end
function integrate(integrator::Implicit, model::ImplicitDynamicsModel, 
                   x::StaticVector{Nx}, u::StaticVector{Nu}, t, h) where {Nx,Nu} 
    cache = getnewtoncache(integrator) 
    newton_iters = cache.newton_iters
    tol = cache.newton_tol

    x = SVector(x)
    u = SVector(u)

    J2 = cache.J2
    J1 = cache.J1
    y2 = cache.y2
    y1 = cache.y1
    z1 = StaticKnotPoint(x, u, t, h)
    z2 = StaticKnotPoint(z1)
    ix = SVector{Nx}(1:Nx) 

    # Use the current state as the current guess
    xn = state(z1)

    diff = default_diffmethod(model.continuous_dynamics)
    for iter = 1:newton_iters
        # Set the guess for the next state
        z2 = setstate(z2, xn)

        # Calculate the residual
        r = dynamics_error(integrator, model.continuous_dynamics, z2, z1)

        # Calculate the Jacobian wrt x2
        dynamics_error_jacobian!(StaticReturn(), diff, model, 
                                 J2, J1, y2, y1, z2, z1)
        A = J2[ix, ix] 

        if norm(r) < tol
            break
        end

        # Get the step
        F = lu(A)
        dx = F \ r
        xn -= dx
    end
    setdata!(cache.z2, getdata(z1))  # copy input into cache for Jacobian check
    return xn
end

function integrate!(integrator::Implicit, model::ImplicitDynamicsModel, xn, 
                    z::AbstractKnotPoint)
    integrate!(integrator, model, xn, state(z), control(z), z.t, z.dt)
end
function integrate!(integrator::Implicit, model::ImplicitDynamicsModel, xn, 
                    x, u, t, h)
    cache = getnewtoncache(integrator) 
    newton_iters = cache.newton_iters
    tol = cache.newton_tol

    z1 = cache.z1
    z2 = cache.z2
    # copyto!(z1, z)
    setstate!(z1, x)
    setcontrol!(z1, u)
    z1.t = t
    z1.dt = h
    copyto!(z2, z1)

    n,m = dims(model)
    J2 = cache.J2
    J1 = cache.J1
    r = cache.y2
    dx = cache.y1
    ipiv = cache.ipiv
    A = cache.A
    Aview = @view J2[:, 1:n]

    # Use the current state as the guess
    copyto!(xn, state(z1))

    diff = default_diffmethod(model.continuous_dynamics)
    for iter = 1:newton_iters
        # Set the guess for the next state
        setstate!(z2, xn)

        # Calculate the residual
        dynamics_error!(integrator, model.continuous_dynamics, r, dx, z2, z1)

        if norm(r) < tol
            break
        end

        # Calculate the Jacobian wrt x2
        dynamics_error_jacobian!(InPlace(), diff, model, J2, J1, r, dx, z2, z1)
        A .= Aview 

        # Get the step
        dx .= r
        F = lu!(A, ipiv)
        ldiv!(F, dx)

        # Apply the step
        xn .-= dx
    end
    setdata!(cache.z2, getdata(z1))  # copy input to cache for Jacobian check
end

# Use Implicit Function Theorem to calculate the dynamics Jacobians
function jacobian!(integrator::Implicit, ::StaticReturn, diff::DiffMethod, 
                   model::ImplicitDynamicsModel, J, y, z::AbstractKnotPoint{Nx,Nu}
                   ) where {Nx,Nu}
    cache = getnewtoncache(integrator) 
    J2 = cache.J2
    J1 = cache.J1
    ix = SVector{Nx}(1:Nx)

    # Update Jacobian
    aresame = maxdiff(cache.z2, z) < √eps()
    if !aresame
        @debug "Solving for next state to get Jacobian using IFT."
        evaluate(model, z)
    else
        @debug "Using cached Static Factorization"
    end
    A2 = J2[ix,ix]
    Jstatic = SMatrix{Nx,Nx+Nu}(J1)
    J .= A2 \ Jstatic
    J .*= -1
    return
end

function jacobian!(integrator::Implicit, ::InPlace, diff::DiffMethod, 
                   model::ImplicitDynamicsModel, J, y, z::AbstractKnotPoint)
    n,m = dims(z)
    cache = getnewtoncache(integrator) 
    J1 = cache.J1

    aresame = maxdiff(cache.z2, z) < √eps()
    local F
    if !aresame
        @debug "Solving for next state to get Jacobian using IFT."
        y .= state(z)
        evaluate!(model, y, z)
    else
        @debug "Using cached Factorization"
    end
    F = cache.F
    J .= J1
    J .*= -1
    ldiv!(F, J)
    return
end

"""
    ImplicitNewtonCache

A cache for the temporary variables needed while solving for the next state 
    using Newton's method for implicit dynamics. Also used to calculate the 
    Jacobians using the implicit function theorem.

Also provides a couple options for controlling the behavior of the Newton solve.

Every implicit integrator should store this cache internally and define it's 
getter method `getnewtoncache(::Implicit)`.
"""
mutable struct ImplicitNewtonCache
    J2::Matrix{Float64}
    J1::Matrix{Float64}
    y2::Vector{Float64}
    y1::Vector{Float64}
    z2::StaticKnotPoint{Any,Any,Vector{Float64},Float64}
    z1::KnotPoint{Any,Any,Vector{Float64},Float64}
    ipiv::Vector{BlasInt}
    A::Matrix{Float64}
    F::LinearAlgebra.LU{Float64, Matrix{Float64}} 
    newton_iters::Int    # number of newton iterations
    newton_tol::Float64  # Newton tolerance
end
function ImplicitNewtonCache(n::Integer, m::Integer)
    J2 = zeros(n,n+m)
    J1 = zeros(n,n+m)
    y2 = zeros(n)
    y1 = zeros(n)
    v = zeros(n+m)
    z2 = StaticKnotPoint{Any,Any}(n, m, v, 0.0, NaN)
    z1 = KnotPoint{Any,Any}(n, m, copy(v), 0.0, NaN)
    ipiv = zeros(BlasInt, n)
    A = zeros(n,n) 
    F = lu!(A, check=false)
    iters = 10    # Default number of Newton iterations
    tol = 1e-12   # Default Newton tolerance
    ImplicitNewtonCache(J2, J1, y2, y1, z2, z1, ipiv, A, F, iters, tol)
end

"""
    ImplicitMidpoint

A symplectic method with second-order accuracy. A great option for those wanting 
good performance with few calls to the dynamics.

```math
x_1 + h f(\\frac{1}{2}(x_1 + x_2), u_1, t + \\frac{1}{2} h) - x_2 = 0
````
"""
struct ImplicitMidpoint <: Implicit
    xmid::ADVector{Float64}
    cache::ImplicitNewtonCache
end
function ImplicitMidpoint(n::Integer, m::Integer)
    cache = ImplicitNewtonCache(n, m)
    ImplicitMidpoint(ADVector{Float64}(n), cache)
end

getnewtoncache(integrator::ImplicitMidpoint) = integrator.cache

function dynamics_error(
    ::ImplicitMidpoint,
    model::ContinuousDynamics,
    z2::AbstractKnotPoint,
    z1::AbstractKnotPoint,
)
    t, h = time(z1), timestep(z1)
    x1, u1 = state(z1), control(z1)
    x2, u2 = state(z2), control(z2)
    xmid = (x1 + x2) / 2
    fmid = dynamics(model, xmid, u1, t + h / 2)
    x1 + h * fmid - x2
end

function dynamics_error!(
    int::ImplicitMidpoint,
    model::ContinuousDynamics,
    y2,
    y1,
    z2::AbstractKnotPoint,
    z1::AbstractKnotPoint,
)
    T = eltype(y2)
    t, h = time(z1), timestep(z1)
    x1, u1 = state(z1), control(z1)
    x2, u2 = state(z2), control(z2)
    xmid = int.xmid[T]::Vector{T}

    @. xmid = (x1 + x2) / 2
    dynamics!(model, y2, xmid, u1, t + h / 2)
    @. y2 = x1 + h * y2 - x2
    return nothing
end

function dynamics_error_jacobian!(
    int::ImplicitMidpoint,
    ::StaticReturn,
    model::ContinuousDynamics,
    J2,
    J1,
    y2,
    y1,
    z2::AbstractKnotPoint,
    z1::AbstractKnotPoint,
)
    n, m = dims(model)
    ix = SVector{n}(1:n)
    iu = SVector{m}(n+1:n+m)
    t, h = time(z1), timestep(z1)
    x1, u1 = state(z1), control(z1)
    x2, u2 = state(z2), control(z2)

    xmid = (x1 + x2) / 2
    jacobian!(model, J1, y1, xmid, u1, t)

    J1 .*= h
    J1[ix,ix] ./= 2
    J2[ix,ix] .= J1[ix,ix]
    J2[ix,iu] .= 0
    for i = 1:n
        J1[i,i] += 1.0
        J2[i,i] -= 1.0
    end
    return nothing
end

function dynamics_error_jacobian!(
    int::ImplicitMidpoint,
    ::InPlace,
    model::ContinuousDynamics,
    J2,
    J1,
    y2,
    y1,
    z2::AbstractKnotPoint,
    z1::AbstractKnotPoint,
)
    n, m = dims(model)
    ix = 1:n 
    iu = n+1:n+m 
    t, h = time(z1), timestep(z1)
    x1, u1 = state(z1), control(z1)
    x2, u2 = state(z2), control(z2)
    xmid = int.xmid[Float64]

    @. xmid = (x1 + x2) / 2
    jacobian!(model, J1, y1, xmid, u1, t)
    J1 .*= h
    A1 = view(J1, ix, ix)
    A2 = view(J2, ix, ix)
    A1 ./= 2
    A2 .= A1
    view(J2, ix, iu) .= 0
    for i = 1:n
        J1[i,i] += 1.0
        J2[i,i] -= 1.0
    end
    return nothing
end