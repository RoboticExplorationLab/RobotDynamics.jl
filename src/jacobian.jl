"""
    DynamicsJacobian{n,nm,T}

Custom `n × (n+m)` matrix specifying a dynamics Jacobian for a forced dynamical system with
`n` states and `m` controls. The Jacobian is structured as `[∂x ∂u]` where `x` and `u` are
the state and control vectors, respectively.

The `DynamicsJacobian` `D` provides access to the partial derivatives `A = ∂x`  and `B = ∂u` via
direct access `D.A` and `D.B`, returning a view into the underlying `Matrix`, or

    RobotDynamics.get_A(D)
    RobotDynamics.get_B(D)

which return an `SMatrix`. Note that this method should be used with caution for systems with
large state and/or control dimensions.

# Constructors
    DynamicsJacobian(model::AbstractModel)
    DynamicsJacobian(n::Int, m::Int)
    DynamicsJacobian(D::StaticMatrix)

where `D` is a `StaticMatrix` of appropriate size. Since `DynamicsJacobian` implements the
`StaticMatrix` interface, is also supports all the constructors and operations inherent to
a `StaticMatrix`.
"""
struct DynamicsJacobian{S1,S2,T} <: StaticMatrix{S1,S2,T}
    data::SizedMatrix{S1,S2,T,2,Matrix{T}}
    A::SubArray{T,2,Matrix{T}, Tuple{Base.Slice{Base.OneTo{Int}}, UnitRange{Int}},true}
    B::SubArray{T,2,Matrix{T}, Tuple{Base.Slice{Base.OneTo{Int}}, UnitRange{Int}},true}
    function DynamicsJacobian(F::StaticMatrix{n,nm}) where {n,nm}
        m = nm-n
        data = SizedMatrix{n,nm}(F)
        A = view(data.data, :, 1:n)
        B = view(data.data, :, n .+ (1:m))
        new{n,nm,eltype(data)}(data, A, B)
    end
end

DynamicsJacobian(n::Int, m::Int) = DynamicsJacobian(SizedMatrix{n,n+m}(zeros(n,n+m)))

@inline DynamicsJacobian{S1,S2}(t::NTuple) where {S1,S2}= DynamicsJacobian(SMatrix{S1,S2}(t))
@inline DynamicsJacobian{S1,S2,T}(t::NTuple) where {S1,S2,T}= DynamicsJacobian(SMatrix{S1,S2,T}(t))
Base.@propagate_inbounds Base.getindex(D::DynamicsJacobian, i::Int) = D.data[i]
Base.@propagate_inbounds Base.setindex!(D::DynamicsJacobian, x, i::Int) = D.data[i] = x
Base.@propagate_inbounds Base.setindex!(D::DynamicsJacobian, x, inds::Vararg{Int64}) =
    setindex!(D.data, x, inds...)
@inline Base.Tuple(D::DynamicsJacobian) = Tuple(D.data)

"""
    get_A(D::DynamicsJacobian)

Return the partial derivative of the dynamics with respect to the state input as an `SMatrix`,
given the full dynamics Jacobian.
"""
@generated function get_static_A(D::DynamicsJacobian{n,nm}) where {n,nm}
    expr = [:(D[$i]) for i = 1:n*n]
    :(SMatrix{$n,$n}(tuple($(expr...))))
end

"""
    get_B(D::DynamicsJacobian)

Return the partial derivative of the dynamics with respect to the control input as an `SMatrix`,
given the full dynamics Jacobian.
"""
@generated function get_static_B(D::DynamicsJacobian{n,nm}) where {n,nm}
    m = nm - n
    inds = (1:n*m) .+ n*n
    expr = [:(D[$i]) for i in inds]
    :(SMatrix{$n,$m}(tuple($(expr...))))
end

get_A(D::DynamicsJacobian) = D.A
get_B(D::DynamicsJacobian) = D.B


@inline get_data(A::AbstractArray) = A
@inline get_data(A::SizedArray) = A.data
@inline get_data(A::DynamicsJacobian) = A.data.data
