using Plots
n,m,N = 6,3,21
times = range(0,pi,length=N)
X = [SA[sin(t), cos(t), t, t^2, sin(t) + t, sqrt(t)] for t in times]
U = [@SVector rand(m) for k = 1:N-1]
Z = RD.SampledTrajectory(X, U, dt=diff(times))

plot(X)
plot(X, inds=1:2)
plot(times, X, inds=1:2)
plot!(times, X, inds=3:n)
plot(times, X)  # should be the same the previous
p = plot(times, X, color=[:black :red :green :blue :yellow :purple])
@test length(p.series_list) == n
@test p.series_list[1].plotattributes[:linecolor] == RGBA(0,0,0)

# Traj
p = traj2(X)
traj2!(-sin.(times), cos.(times))
@test length(p.series_list) == 2

p = traj2(Z, xind=3, yind=5, linewidth=2, label="something")
@test length(p.series_list) == 1

p = plot(Z)
@test length(p.series_list) == 9

p = plot(Z, inds=3:6)
@test length(p.series_list) == 4 