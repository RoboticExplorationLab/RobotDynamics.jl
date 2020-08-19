Random.seed!(10)
for i = 1:20
    local n = rand(15:40)
    local m = rand(5:15)
    tol = randn()
    A,B = gencontrollable(n,m,tol=tol)
    @test iscontrollable(A,B)
    @test isstabled(A) == (tol >= 0)
end

for i = 1:20
    local n = rand(15:30)
    local m = rand(5:15)
    tol = randn()
    A,B = gencontrollable(n,m,:continuous,tol=tol,maxiter=500)
    @test iscontrollable(A,B)
    @test isstablec(A) == (tol >= 0)
end