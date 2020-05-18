struct DynamicsJacobian{S1,S2,T} <: StaticMatrix{S1,S2,T}
    data::SizedMatrix{S1,S2,T,2}
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

@inline get_data(A::AbstractArray) = A
@inline get_data(A::SizedArray) = A.data
@inline get_data(A::DynamicsJacobian) = A.data.data
