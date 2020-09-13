abstract type PassThrough <: QuadratureRule end

struct LinearModel{n,m,T} <: AbstractModel
    A::Vector{SizedMatrix{n,n,T,2}}
    B::Vector{SizedMatrix{n,m,T,2}}
    d::Vector{SizedVector{n,T,1}}
    times::Vector{T}
    dt::T
    xdot::MVector{n,T}
    use_static::Bool
    function LinearModel(
            A::Vector{TA},
            B::Vector{TB},
            d::Vector{Td},
            times::AbstractVector;
            dt::Real=0,
            use_static::Bool=(length(A[1]) < 14*14)
        ) where {TA <: AbstractMatrix,TB<:AbstractMatrix,Td<:AbstractVector}
        n,m = size(B[1])
        @assert size(A[1]) == (n,n)
        isempty(d) || @assert size(d[1]) == (n,)
        @assert length(A) == length(B)
        @assert length(A) > 0
        @assert issorted(times)
        T = promote_type(eltype(TA), eltype(TB), eltype(Td))
        A = SizedMatrix{n,n,T}.(a for a in A)
        B = SizedMatrix{n,m,T}.(b for b in B)
        d = SizedVector{n,T}.(d_ for d_ in d)
        times = Vector{T}(times)
        xdot = @MVector zeros(n)
        new{n,m,T}(A, B, d, times, dt, xdot, use_static)
    end
end
state_dim(::LinearModel{n}) where n = n
control_dim(::LinearModel{<:Any,m}) where m = m

is_discrete(model::LinearModel) = model.dt !== zero(model.dt)
is_affine(model::LinearModel) = !isempty(model.d)
is_timevarying(model::LinearModel) = !isempty(model.times)
get_k(model::LinearModel, t) = is_timevarying(model) ? searchsortedlast(model.times, t) : 1

function LinearModel(A::AbstractMatrix, B::AbstractMatrix, 
        d=SVector{size(A,1),eltype(A)}[]; 
        dt=0, kwargs...)
        times = 1:0
    LinearModel([A],[B], d, times, dt=dt; kwargs...)
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
    dt_model = isnan(model.dt) ? model.times[k+1] - model.times[k] : model.dt
    @assert dt ≈ dt_model "Incorrect dt. Expected $dt_model, got $dt."
    linear_dynamics(model, x, u, k)
end

