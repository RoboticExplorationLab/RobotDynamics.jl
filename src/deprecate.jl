@deprecate state_diff_jacobian!(args...) errstate_jacobian!(args...)
@deprecate ∇²differential!(args...) ∇errstate_jacobian!(args...)
@deprecate SampledTrajectory(X,U,dt::Vector{<:Real}) SampledTrajectory(X,U,dt=dt)
@deprecate SampledTrajectory(x::AbstractVector{<:Real}, u::AbstractVector{<:Real}, dt::AbstractVector{<:Real}) Traj([copy(x) for k = 1:length(dt)], [copy(u) for k = 1:length(dt)-1], dt=dt)
@deprecate get_data(x::SampledTrajectory) getdata(x)
Base.Base.@deprecate_binding Traj SampledTrajectory true

import Base.size
@deprecate size(fun::AbstractFunction) dims(fun)
@deprecate traj_size(Z) dims(Z)