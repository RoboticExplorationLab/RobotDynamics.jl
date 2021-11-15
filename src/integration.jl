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

########################################
# Explicit Methods
########################################

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

function jacobian!(int::Euler, sig::FunctionSignature, model, J, xn, x, u, t, h)
    # Call the user-defined Continuous-time Jacobian
    jacobian!(model, J, xn, x, u, t)
    J .*= h
    for i = 1:state_dim(model)
        J[i, i] += 1.0
    end
    return nothing
end

"Fourth-order Runge-Kutta method with zero-order-old on the controls"
struct RK4 <: Explicit
    k1::ADVector{Float64}
    k2::ADVector{Float64}
    k3::ADVector{Float64}
    k4::ADVector{Float64}
    A::Vector{Matrix{Float64}}
    B::Vector{Matrix{Float64}}
    dA::Vector{Matrix{Float64}}
    dB::Vector{Matrix{Float64}}
    function RK4(n::Integer, m::Integer)
        k1, k2 = ADVector{Float64}(n), ADVector{Float64}(n)
        k3, k4 = ADVector{Float64}(n), ADVector{Float64}(n)
        A = [zeros(n, n) for i = 1:4]
        B = [zeros(n, m) for i = 1:4]
        dA = [zeros(n, n) for i = 1:4]
        dB = [zeros(n, m) for i = 1:4]
        new(k1, k2, k3, k4, A, B, dA, dB)
    end
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
    k1 .= 2 .* x
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
    n, m = size(model)
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
    n, m = size(model)
    ix, iu = 1:n, n+1:n+m

    dynamics!(model, k1, x, u, t)
    jacobian!(model, J, xn, x, u, t)
    A1 .= @view J[ix, ix]
    B1 .= @view J[ix, iu]

    @. xn = x + k1 * h / 2
    dynamics!(model, k2, xn, u, t + h / 2)
    jacobian!(model, J, xn, x, u, t + h / 2)
    A2 .= @view J[ix, ix]
    B2 .= @view J[ix, iu]

    @. xn = x + k2 * h / 2
    dynamics!(model, k3, xn, u, t + h / 2)
    jacobian!(model, J, xn, x, u, t + h / 2)
    A3 .= @view J[ix, ix]
    B3 .= @view J[ix, iu]

    @. xn = x + k3 * h
    jacobian!(model, J, xn, x, u, t + h)
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

struct ImplicitMidpoint <: Implicit
    xmid::ADVector{Float64}
    function ImplicitMidpoint(n::Integer, m::Integer)
        new(ADVector{Float64}(n))
    end
end

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
    n, m = size(model)
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
    n, m = size(model)
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

a = 1