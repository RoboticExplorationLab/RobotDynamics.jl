struct NotImplementedError <: Exception 
    msg::String
end

function Base.showerror(io::IO, err::NotImplementedError)
    print(io, "NotImplementedError: ")
    print(io, err.msg)
end

"""
    maxdiff(a, b)

Get the maximum difference between vectors. Equivalent to the infinity norm of the 
difference, but avoids forming the temporary difference vector.
"""
function maxdiff(a, b)
    err = -Inf
    for i = min(length(a), length(b))
        err = max(err, abs(a[i] - b[i]))
    end
    return err
end

# Add non-allocating lu! method that allows you to pass in the pivot vector
using LinearAlgebra: BlasFloat, BlasInt
using LinearAlgebra.BLAS: @blasfunc
@static if VERSION < v"1.7"
    const libblas = LinearAlgebra.LAPACK.liblapack
else
    const libblas = LinearAlgebra.LAPACK.libblastrampoline 
end
for (getrf,elty) in ((:dgetrf_, :Float64), (:sgetrf, :Float32))
    @eval begin
        function LinearAlgebra.LAPACK.getrf!(A::AbstractMatrix{$elty}, ipiv::AbstractVector{BlasInt})
            LinearAlgebra.require_one_based_indexing(A)
            LinearAlgebra.chkstride1(A)
            m, n = size(A)
            @assert length(ipiv) == min(m, n)
            lda  = max(1,stride(A, 2))
            info = Ref{BlasInt}()
            ccall((@blasfunc($getrf), libblas), Cvoid,
                    (Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty},
                    Ref{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt}),
                    m, n, A, lda, ipiv, info)
            LAPACK.chkargsok(info[])
            A, ipiv, info[] #Error code is stored in LU factorization type
        end
    end
end
function LinearAlgebra.lu!(A::StridedMatrix{T}, ipiv::AbstractVector{BlasInt}; check::Bool=true) where T
    lpt = LAPACK.getrf!(A, ipiv)
    check && LinearAlgebra.checknonsingular(lpt[3])
    return LU{T,typeof(A)}(lpt[1], lpt[2], lpt[3])
end

function get_dependency_version(pkg_name)
    m = Pkg.dependencies()
    v = m[findfirst(v -> v.name == pkg_name, m)].version
    v
end