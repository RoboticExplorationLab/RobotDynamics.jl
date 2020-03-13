# export
# 	DynamicsExpansion

struct DynamicsExpansion{T,N,N̄,M}
	∇f::Matrix{T} # n × (n+m+1)
	A_::SubArray{T,2,Matrix{T},Tuple{UnitRange{Int},UnitRange{Int}},false}
	B_::SubArray{T,2,Matrix{T},Tuple{UnitRange{Int},UnitRange{Int}},false}
	A::SizedMatrix{N̄,N̄,T,2}
	B::SizedMatrix{N̄,M,T,2}
	tmpA::SizedMatrix{N,N,T,2}
	tmpB::SizedMatrix{N,M,T,2}
	tmp::SizedMatrix{N,N̄,T,2}
	function DynamicsExpansion{T}(n0::Int, n::Int, m::Int) where T
		∇f = zeros(n0,n0+m)
		ix = 1:n0
		iu = n0 .+ (1:m)
		A_ = view(∇f, ix, ix)
		B_ = view(∇f, ix, iu)
		A = SizedMatrix{n,n}(zeros(n,n))
		B = SizedMatrix{n,m}(zeros(n,m))
		tmpA = SizedMatrix{n0,n0}(zeros(n0,n0))
		tmpB = SizedMatrix{n0,m}(zeros(n0,m))
		tmp = zeros(n0,n)
		new{T,n0,n,m}(∇f,A_,B_,A,B,tmpA,tmpB,tmp)
	end
	function DynamicsExpansion{T}(n::Int, m::Int) where T
		∇f = zeros(n,n+m)
		ix = 1:n
		iu = n .+ (1:m)
		A_ = view(∇f, ix, ix)
		B_ = view(∇f, ix, iu)
		A = SizedMatrix{n,n}(zeros(n,n))
		B = SizedMatrix{n,m}(zeros(n,m))
		tmpA = A
		tmpB = B
		tmp = zeros(n,n)
		new{T,n,n,m}(∇f,A_,B_,A,B,tmpA,tmpB,tmp)
	end
end
