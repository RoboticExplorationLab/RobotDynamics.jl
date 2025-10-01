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