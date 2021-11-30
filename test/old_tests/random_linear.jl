using LinearAlgebra


function isstabled(A)
    all(x->abs(x)<=1, eigvals(A))
end

function isstablec(A)
    all(x->x<=0, eigvals(A))
end

function controllability(A,B)
    n,m = size(B)
    R = zeros(n,n*m)
    Ak = zero(A) .+ I(n)
    for k = 1:n
        Rk = view(R, :, (k-1)*m .+ (1:m))
        Rk .= Ak*B
        Ak = Ak * A
    end
    return R
end

iscontrollable(A,B) = rank(controllability(A,B)) == size(A,1)

function genA(v, n, m)
    # Generate a random orthogonal matrix
    X = randn(n,n)
    Q = qr(X).Q
    A = Q*Diagonal(v)*Q'
end

genB(n,m) = randn(n,m)

function gendiscrete(n, m, tol::Real=1e-4)
    v = randn(n) 
    v = v ./ (norm(v,Inf) + tol)     # make it (marginally) discrete stable
    A = genA(v, n, m)
    B = genB(n,m)
    A,B
end

function gencontinuous(n, m, tol::Real=1e-4)
    v = randn(n) 
    v = v .- (maximum(v) + tol)     # make it (marginally) continuous stable
    A = genA(v, n, m)
    B = genB(n,m)
    A,B
end

function gencontrollable(n, m, type::Symbol=:discrete; tol::Real=1e-4, maxiter=20)
    cnt = 0
    while true
        if type == :discrete
            A,B = gendiscrete(n, m, tol)
        elseif type == :continuous
            A,B = gencontinuous(n, m, tol)
        end
        iscontrollable(A,B) && (return A,B)
        if cnt > maxiter 
            throw(ErrorException("exceeded max number of attempt to find a controllable system."))
        end
        cnt += 1
    end
end
