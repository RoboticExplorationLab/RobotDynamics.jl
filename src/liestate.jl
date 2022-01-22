"""
	LieGroupModel <: AbstractModel

Abstraction of a dynamical system whose state contains at least one arbitrary rotation.
"""
abstract type LieGroupModel <: ContinuousDynamics end
const DiscreteLieDynamics = DiscretizedDynamics{L,Q} where {L<:LieGroupModel, Q<:QuadratureRule}
statevectortype(::Type{<:LieGroupModel}) = RotationState() 

import Rotations.params

params(::Type{<:UnitQuaternion}) = 4
params(::Type{<:RodriguesParam}) = 3
params(::Type{<:MRP}) = 3
params(::Type{<:RotMatrix3}) = 9
params(::Type{<:RotMatrix2}) = 4

"""
    LieState{R,P}

Specifies a state with rotational components mixed in with standard vector components. All
rotational components are assumed to be parameterizations of 3D rotations.

# Parameters
- `R <: Rotation` is the rotational representation used in the state vector. Must have
`params(::Type{R})` defined, which returns the number of parameters used by the rotation,
as well a constructor that takes each parameter as a separate scalar argument.

- `P <: Tuple{Vararg{Int}}` is a tuple of integers specifying the partitioning of the state
vector. Each element of `P` specifies the length of the vector component between the rotational
components, and `P[1]` and `P[end]` specify the number of vector states at the beginning and
end of the state vector.

# Examples
If we want to construct a state vector like the following: `[v3, q, v2, q, v3]` where `v2`
and `v3` and vector components of length 2 and 3, respectively, and `q` is a 4-dimensional
unit quaternion. The `LieState` for this state vector would be
`LieState{UnitQuaternion{Float64},3,2,3}`. The length should be (3+4+2+4+3) = 16, which can
be verified by `length(s::LieState)`.

# Constructors
    LieState(::Type{R}, P::Tuple{Vararg{Int}})
    LieState(::Type{R}, p1::Int, p2::Int, p3::Int...)
"""
struct LieState{R,P}
    function LieState{R,P}() where {R<:Rotation,P}
        new{R,P::Tuple{Vararg{Int}}}()
    end
end

@inline LieState(::Type{R}, P::Tuple{Vararg{Int}}) where R <: Rotation = LieState{R,P}()
@inline LieState(::Type{R}, P::Int...) where R <: Rotation = LieState{R,P}()
LieState(model::DiscreteLieDynamics) = LieState(model.continuous_dynamics)

"""
    QuatState(n::Int, Q::StaticVector{<:Any,Int})
    QuatState(n::Int, Q::NTuple{<:Any,Int})

Create a `n`-dimensional `LieState` assuming `R = UnitQuaternion{Float64}` and `Q[i]` is the
first index of each quaternion in the state vector.

# Example
If we want to construct a state vector like the following: `[v3, q, v2, q, v3]` where `v2`
and `v3` and vector components of length 2 and 3, respectively, and `q` is a 4-dimensional
unit quaternion. Since the first quaternion starts at index 4, and the second starts at index
10, Q = [4,10]. The entire length of the vector is `n = 16 = 3 + 4 + 2 + 4 + 3`, so we would
call `QuatState(16, SA[4,10])`.
"""
@generated function QuatState(n::Int, Q::StaticVector{NR,Int}) where {NR}
    if NR == 1
        P0 = SVector{0,Int}()
    else
        diffs = [:(Q[$i] - Q[$(i-1)] - 4) for i = 2:NR]
        P0 = :(SVector{NR-1}(tuple($(diffs...))))
    end
    quote
        P = $P0
        P = push(P, n - (Q[end] + 4) + 1)
        P = pushfirst(P, Q[1] - 1)
        return LieState(UnitQuaternion{Float64}, Tuple(P))
    end
end
@inline QuatState(n::Int, Q::NTuple{<:Any,Int}) = QuatState(n, SVector(Q))

"number of rotations"
num_rotations(::LieState{<:Any,P}) where P = length(P) - 1

state_dim_vec(::LieState{<:Any,P}) where P = sum(P)
state_dim_rot(s::LieState{R}) where R = params(R)*num_rotations(s)

Base.length(s::LieState{R,P}) where {R,P} = params(R)*num_rotations(s) + sum(P)
Base.length(::Type{LieState{R,P}}) where {R,P} = params(R)*(length(P)-1) + sum(P)

errstate_dim(s::LieState{R,P}) where {R,P} = 3*num_rotations(s) + sum(P)

# Useful functions for meta-programming
rot_inds(R,P, i::Int) = (sum(P[1:i]) + (i-1)*params(R)) .+ (1:params(R))
vec_inds(R,P, i::Int) =
    ((i > 1 ? sum(P[1:i-1]) : 0) + (i-1)*params(R)) .+ (1:P[i])
inds(R,P, i::Int) = isodd(i) ? vec_inds(R,P, 1+i÷2) : rot_inds(R,P, i÷2)
@inline inds(s::LieState{R,P}, i::Int) where {R,P} = inds(R, P, i)
rot_state(R,P, i::Int, sym=:x) = [:($(sym)[$j]) for j in rot_inds(R,P,i)]

"""
    vec_states(model::LieGroupModel, x)
    vec_states(s::LieState, x)

Extracts the "vector" states out of the state vector `x` for a `LieGroupModel`. Returns
a tuple `v` of `SVector`s, where `length(v[i])` is equal to the length specified by the
`LieState`.
"""
@generated function vec_states(s::LieState{R,P}, x) where {R,P}
    T = eltype(x)
    inds = [vec_inds(R,P,i) for i = 1:length(P)]
    states = [[:(x[$i]) for i in ind] for ind in inds]
    vecs = [:(SVector{$(length(inds)),$T}($(inds...))) for inds in states]
    quote
        tuple($(vecs...))
    end
end
@inline vec_states(model::LieGroupModel, x) = vec_states(LieState(model), x)

"""
    vec_states(model::LieGroupModel, x)
    vec_states(s::LieState, x)

Extracts the rotations out of the state vector `x` for a `LieGroupModel`. Returns
a tuple rotations, whose type matches the rotation type specified in the `LieState`.
"""
@generated function rot_states(s::LieState{R,P}, x) where {R,P}
    T = eltype(x)
    inds = [rot_inds(R,P,i) for i = 1:length(P)-1]
    states = [[:(x[$i]) for i in ind] for ind in inds]
    if R <: UnitQuaternion
        vecs = [:(R($(inds...),false)) for inds in states]
    else
        vecs = [:(R($(inds...))) for inds in states]
    end
    quote
        tuple($(vecs...))
    end
end
@inline rot_states(model::LieGroupModel, x) = rot_states(LieState(model), x)

@generated function Base.rand(s::LieState{R,P}) where {R,P}
    nr = length(P) - 1   # number of rotations
    np = nr + length(P)  # number of partitions
    n = length(s)

    q = [:($(Symbol("q$i")) = Rotations.params(rand(R))) for i = 1:nr]

    x = Expr[]
    for i = 1:np
        if isodd(i)  # vector part
            vi = inds(R,P,i)
            for j in vi
                push!(x, :(rand()))
            end
        else
            r = i ÷ 2
            ri = inds(R,P,i)
            for j = 1:Rotations.params(R)
                push!(x, :($(Symbol("q$r"))[$j]))
            end
        end
    end
    quote
        $(Expr(:block, q...))
        $(:(SVector{$n}(tuple($(x...)))))
    end
end

function Base.rand(model::LieGroupModel)
    rand(LieState(model)), @SVector rand(control_dim(model))
end

@inline state_dim(model::LieGroupModel) = length(LieState(model))

@inline state_diff!(::RotationState, model::AbstractModel, dx::AbstractVector,
    x::AbstractVector, x0::AbstractVector, errmap=Rotations.CayleyMap()) =
    state_diff!(LieState(model), dx, x, x0, errmap)
@inline state_diff(::RotationState, model::AbstractModel, x::AbstractVector, 
    x0::AbstractVector, errmap=Rotations.CayleyMap()) =
    state_diff(LieState(model), x, x0, errmap)
@inline errstate_dim(::RotationState, model::AbstractModel) = errstate_dim(LieState(model))

function _state_diff_expr(R,P)
    nr = length(P) - 1   # number of rotations
    np = nr + length(P)  # number of partitions
    n̄ = 3*nr + sum(P)    # error state size

    # Generate a vector of δq = q0\q expressions for each q in the state
    dq = [:($(Symbol("q$i")) = Rotations.rotation_error(
        R($(rot_state(R,P,i)...)), R($(rot_state(R,P,i,:x0)...)), errmap)) for i = 1:nr
    ]

    # Generate the vector of expressions for each element of the state differential
    dx = Expr[]
    for i = 1:np
        # for the vector parts, simply subtract the indices from the original states
        if isodd(i)
            vi = inds(R,P,i)
            for j in vi
                push!(dx, :(x[$j] - x0[$j]))
            end
        # for the rotational parts, use the elements of the rotational errors generated above
        else
            r = i ÷ 2  # rotation index
            for j = 1:3
                push!(dx, :($(Symbol("q$r"))[$j]))
            end
        end
    end
    return dq,dx
end
@generated function state_diff(s::LieState{R,P}, x::AbstractVector, x0::AbstractVector, 
        errmap=Rotations.CayleyMap()) where {R,P}
    nr = length(P) - 1   # number of rotations
    n̄ = 3*nr + sum(P)    # error state size
    dq,dx = _state_diff_expr(R,P)
    quote
        $(Expr(:block, dq...))
        $(:(SVector{$n̄}(tuple($(dx...)))))
    end
end
@generated function state_diff!(s::LieState{R,P}, dx::AbstractVector, x::AbstractVector, 
        x0::AbstractVector, errmap=Rotations.CayleyMap()) where {R,P}
    dq,dx = _state_diff_expr(R,P)
    dx_expr = map(enumerate(dx)) do (i,e)
        :(dx[$i] = $e) 
    end
    quote
        $(Expr(:block, dq...))
        $(Expr(:block, dx_expr...))
        return
    end
end

@inline state_diff_jacobian!(::RotationState, model::AbstractModel, G, x) =
    state_diff_jacobian!(LieState(model), G, x)

@generated function state_diff_jacobian!(s::LieState{R,P}, G, x) where {R,P}
    nr = length(P) - 1   # number of rotations
    np = nr + length(P)  # number of partitions
    nv = length(P)
    n̄ = 3*nr + sum(P)    # error state size
    n = params(R)*nr + sum(P)

    # Generate a vector of δq = q0\q expressions for each q in the state
    q = [:(R($(rot_state(R,P,i)...))) for i = 1:nr]

    # Generate a vector of expressions assigning 1s to all the vector state diagonals
    Gv = Expr[]
    r = 1
    c = 1
    for k = 1:nv
        for j = 1:P[k]
            push!(Gv, :(G[$(LinearIndices((n,n̄))[r,c])] = 1))
            r += 1
            c += 1
        end
        r += params(R)
        c += 3
    end

    # Generate a vector of expressions assigning the differential Jacobians for the rotations
    Gr = Expr[]
    for k = 1:nr
        rinds = rot_inds(R, P, k)
        cinds = (sum(P[1:k]) + 3*(k-1)) .+ (1:3)
        push!(Gr, :(G[$rinds,$cinds] .= Rotations.∇differential($(q[k]))))
    end
    quote
        $(Expr(:block, Gv...))
        $(Expr(:block, Gr...))
        return nothing
    end
end

@inline ∇²differential!(::RotationState, model::AbstractModel, ∇G, x::StaticVector, dx::AbstractVector) =
    ∇²differential!(LieState(model), ∇G, x, dx)
@generated function ∇²differential!(s::LieState{R,P}, ∇G, x::StaticVector, dx::AbstractVector) where {R,P}
    nr = length(P) - 1   # number of rotations
    np = nr + length(P)  # number of partitions
    nv = length(P)
    n̄ = 3*nr + sum(P)    # error state size
    n = params(R)*nr + sum(P)

    # Generate a vector of δq = q0\q expressions for each q in the state
    q  = [:(R($(rot_state(R,P,i,:x)...)))  for i = 1:nr]
    dq = [:(SVector{$(params(R))}($(rot_state(R,P,i,:dx)...))) for i = 1:nr]

    Gr = Expr[]
    for k = 1:nr
        cinds = (sum(P[1:k]) + 3*(k-1)) .+ (1:3)
        push!(Gr, :(view(∇G,$cinds,$cinds) .= SMatrix{3,3}(Rotations.∇²differential($(q[k]), $(dq[k])))))
    end
    quote
        $(Expr(:block, Gr...))
        return nothing
    end
end
