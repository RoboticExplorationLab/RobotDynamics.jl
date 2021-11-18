const VecVec = Vector{<:StaticVector{<:Any,<:Real}}

# This overwrites the default behavior for plotting vectors of static vectors
#   provides convenient methods for plotting state and control trajectories
#   plot(X; inds=1:3) will plot the first three states of the trajectory X 
#   can also pass time in as the first argument, e.g. plot(t,X, inds=2:3) 
@recipe function f(::Type{T}, X::T) where T <: VecVec
    hcat(Vector.(X)...)'
end

@recipe function f(X::VecVec; inds=1:length(X[1]))
    (hcat(Vector.(X)...)[inds,:])'
end

@recipe function f(t::AbstractVector{<:Real}, X::VecVec; inds=1:length(X[1]))
    t, (hcat(Vector.(X)...)[inds,:])'
end

@recipe function f(Z::T; inds=1:state_dim(Z) + control_dim(Z)) where T <: AbstractTrajectory
    default_xlabels = ["x" * string(i) for i = 1:state_dim(Z)]
    default_ulabels = ["u" * string(i) for i = 1:control_dim(Z)]
    default_labels = [default_xlabels; default_ulabels]
    xguide --> "time (s)"
    yguide --> "states"
    label --> reshape(default_labels[inds], 1, :)
    gettimes(Z), (hcat(Vector.(get_data(Z))...)[inds,:])'
end

"""
    traj2(x, y)
    traj2(X; xind=1, yind=2)
    traj2(Z::AbstractTrajectory; xind=1, yind=2)

Plot a 2D state trajectory, given the x and y positions of the robot. If given
a state trajectory, the use optional `xind` and `yind` arguments to provide the
location of the x and y positions in the state vector (typically 1 and 2, respectively).
"""
@userplot Traj2

@recipe function f(traj::Traj2; xind=1, yind=2)
    # Process input
    #   Input a vector of static vectors
    if length(traj.args) == 1 && (traj.args[1] isa VecVec || traj.args[1] isa AbstractTrajectory)
        if traj.args[1] isa VecVec
            X = traj.args[1]
        else
            X = states(traj.args[1])
        end
        xs = [x[xind] for x in X]
        ys = [x[yind] for x in X]
    #   Input the x and y vectors independently
    elseif length(traj.args) == 2 && 
            traj.args[1] isa AbstractVector{<:Real} &&
            traj.args[2] isa AbstractVector{<:Real}
        xs = traj.args[1]
        ys = traj.args[2]
    else
        throw(ArgumentError("Input must either be a Vector of StaticVector's, or two Vectors of positions"))
    end
    # Defaults
    xguide --> "x"
    yguide --> "y"
    label --> :none
    # Plot x,y
    (xs,ys)
end