"Integration type for systems with user defined discrete dynamics."
abstract type PassThrough <: Explicit end
integrate(::Type{PassThrough}, model::AbstractModel, x::StaticVector, u::StaticVector, t, dt) = x

"Exponential integration for linear systems with ZOH on controls."
abstract type Exponential <: Explicit end

abstract type AbstractLinearModel <: AbstractModel end

"""
    LinearModel{n,m,T} <: AbstractModel

A concrete type for creating efficient linear model representations. This model type will
automatically define the continuous or discrete version of the dynamics and jacobian functions.
Supports continuous/discrete, time invariant/varying, and affine models.

# Constructors
    LinearModel(A::AbstractMatrix, B::AbstractMatrix, [dt=0, use_static]) # time invariant
    LinearModel(A::AbstractMatrix, B::AbstractMatrix, d::AbstractVector, [dt=0, use_static]) # time invariant affine
    LinearModel(A::Vector{TA}, B::Vector{TB}, [times::AbstractVector, dt::Real=0, use_static]) # time varying
    LinearModel(A::Vector{TA}, B::Vector{TB}, d::Vector{Td}, [times::AbstractVector, dt=0, use_static]) # time varying affine

    LinearModel(n::Integer, m::Integer, [is_affine=false, times=1:0, dt=0, use_static]) # constructor with zero dynamics matrices

By default, the model is assumed to be continuous unless a non-zero dt is specified. For time varying models, `searchsortedlast` is 
called on the `times` vector to get the discrete time index from the continuous time. The `use_static` keyword is automatically
specified based on array size, but can be turned off in case of excessive compilation times.
"""
struct LinearModel{n,m,T,e} <: AbstractLinearModel
    A::Vector{SizedMatrix{n,n,T,2,Matrix{T}}}
    B::Vector{SizedMatrix{n,m,T,2,Matrix{T}}}
    d::Vector{SizedVector{n,T,Vector{T}}}
    times::Vector{T}
    dt::T
    xdot::MVector{n,T}
    E::SizedMatrix{e,e,T,2,Matrix{T}}   # Matrix for exponential integration, if needed
    use_static::Bool
    function LinearModel(
            A::Vector{TA},
            B::Vector{TB},
            d::Vector{Td} = SVector{size(A[1],1),eltype(A[1])}[];
            times::AbstractVector = 1:0,
            dt::Real = 0,
            use_static::Bool = (length(A[1]) < 14*14)
        ) where {TA <: AbstractMatrix,TB<:AbstractMatrix,Td<:AbstractVector}
        n,m = size(B[1])
        @assert size(A[1]) == (n,n)
        isempty(d) || @assert size(d[1]) == (n,)
        @assert length(A) == length(B)
        length(A) > 1 && @assert length(A) == length(times) "Must pass in time vector of appropriate length. Expected length of $(length(A)), got $(length(times))." 
        @assert length(A) > 0
        @assert issorted(times)
        T = promote_type(eltype(TA), eltype(TB), eltype(Td))
        A = SizedMatrix{n,n,T}.(a for a in A)
        B = SizedMatrix{n,m,T}.(b for b in B)
        d = SizedVector{n,T}.(d_ for d_ in d)
        times = Vector{T}(times)
        xdot = @MVector zeros(n)

        # size of matrix exponential
        is_affine = !isempty(d)
        e = is_affine ? 2n+m : n+m
        E = SizedMatrix{e,e}(zeros(e,e))

        new{n,m,T,e}(A, B, d, times, dt, xdot, E, use_static)
    end
end
state_dim(::LinearModel{n}) where n = n
control_dim(::LinearModel{<:Any,m}) where m = m

is_discrete(model::LinearModel) = model.dt !== zero(model.dt)
is_affine(model::LinearModel) = !isempty(model.d)
is_timevarying(model::LinearModel) = !isempty(model.times)
get_k(model::LinearModel, t) = is_timevarying(model) ? searchsortedlast(model.times, t) : 1
get_A(model::LinearModel, k=1) = model.A[k]
get_B(model::LinearModel, k=1) = model.B[k]
get_d(model::LinearModel, k=1) = model.d[k]


LinearModel(A::AbstractMatrix, B::AbstractMatrix; dt=0, kwargs...) = LinearModel([A],[B], dt=dt; kwargs...)
LinearModel(A::AbstractMatrix, B::AbstractMatrix, d::AbstractVector; dt=0, kwargs...) = LinearModel([A],[B],[d], dt=dt; kwargs...)

function LinearModel(n::Integer, m::Integer; is_affine=false, times=1:0, kwargs...)
    N_ = (length(times) > 1) ? length(times) : 1

    # only linearize about N-1 points in trajectory 
    A = [zero(SizedMatrix{n,n}) for i=1:N_]
    B = [zero(SizedMatrix{n,m}) for i=1:N_]
    d = is_affine ? [zero(SizedVector{n,Float64}) for i=1:N_] : SizedVector{n,Float64}[]

    LinearModel(A, B, d; times=times, kwargs...)
end

function linear_dynamics(model::LinearModel, x, u, k::Int=1)
    if model.use_static
        A = SMatrix(model.A[k])
        B = SMatrix(model.B[k])
        xdot = linear_dynamics(A, B, x, u)
    else
        A = model.A[k]
        B = model.B[k]
        linear_dynamics!(model.xdot, A, B, x, u)
        xdot = SVector(model.xdot)
    end

    if is_affine(model)
        d = SVector(model.d[k])
        xdot += d
    end

    return xdot
end

linear_dynamics(A, B, x, u) = A*x + B*u
function linear_dynamics!(xdot, A, B, x, u)
    mul!(xdot, A, x)
    mul!(xdot, B, u, 1.0, 1.0)
end

function dynamics(model::LinearModel, x, u, t=0.0)
    @assert !is_discrete(model) "Can't call continuous dynamics on a discrete LinearModel"
    k = get_k(model, t)
    linear_dynamics(model, x, u, k)
end

function discrete_dynamics(::Type{PassThrough}, model::LinearModel, x, u, t, dt)
    @assert is_discrete(model) "Can't call discrete dynamics without integration on a continuous LinearModel"
    k = get_k(model, t)
    dt_model = is_timevarying(model) ? model.times[k+1] - model.times[k] : model.dt
    @assert dt ≈ dt_model "Incorrect dt. Expected $dt_model, got $dt."
    linear_dynamics(model, x, u, k)
end

function jacobian!(∇f::AbstractMatrix, model::LinearModel, z::AbstractKnotPoint)
    @assert !is_discrete(model) "Can't call continuous jacobian on a discrete LinearModel"

	t = z.t
    k = get_k(model, t)

    n = state_dim(model)
    m = control_dim(model)

    ∇f[1:n, 1:n] .= model.A[k]
    ∇f[1:n, (n+1):(n+m)] .= model.B[k]
    true
end

function discrete_jacobian!(::Type{PassThrough}, ∇f, model::LinearModel, z::AbstractKnotPoint{<:Any,n,m}) where {n,m}
    @assert is_discrete(model) "Can't call discrete jacobian without integration on a continuous LinearModel"

    t = z.t
    k = get_k(model, t)
    ix = 1:n
    iu = n .+ (1:m)
    ∇f[ix,ix] .= model.A[k]
    ∇f[ix,iu] .= model.B[k]

    nothing
end


# Exponential Integration
function discrete_dynamics(::Type{Exponential}, model::LinearModel, 
        z::AbstractKnotPoint{<:Any,n,m}) where {n,m}
    k = get_k(model, z.t)
    A = model.A[k]
    B = model.B[k]
    d = model.d[k]
    E = model.E

    # Calculate the matrix exponential
    matrix_exponential!(E, A, B, z)

    if is_affine
        z̄ = [z.z; d]
        return E * z̄
    else
        return E * z.z
    end
end

function discrete_jacobian!(::Type{Exponential}, F, model::LinearModel{n,m},
        z::AbstractKnotPoint) where {n,m}
    # Calculate the matrix exponential
    matrix_exponential!(model, z)

    F .= model.E[1:n, 1:n+m]
end

function matrix_exponential!(E::SizedMatrix, A, B, dt, 
        ::Val{S}=Val(length(E) < 14*14)) where S

    # Copy Jacobian to matrix exponential
    n,m = size(B)
    ix,iu = 1:n, n .+ (1:m)
    E[ix, ix] .= A
    E[ix, iu] .= B

    if size(E,1) == 2n+m
        E[1:n, (1:n) .+ (n+m)] .= I(n)
    end

    # Exponentiate
    E.data .*= dt
    if S
        Ec = SMatrix(E)
    else
        Ec = E.data
    end
    Ed = exp(Ec)

    # Copy back to E
    E .= Ed
end

function discretize!(::Type{Exponential}, F::DynamicsJacobian, d, 
        model::AbstractModel, z::AbstractKnotPoint, is_affine::Bool=true) where Q <: Explicit

end