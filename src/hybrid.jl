abstract type AbstractHybridModel <: AbstractModel end

# Interface
# state_dim(::AbstractHybridModel, k::Int)
# control_dim(::AbstractHybridModel, k::Int)
# get_times(::AbstractHybridModel)

# Implemented functions
get_k(model::AbstractHybridModel, t::Real) = searchsortedlast(get_times(model), t)

struct HybridModel <: AbstractHybridModel
    models::Vector{AbstractModel}
    model_inds::Vector{Int}
    times::Vector{Float64}
    n::Vector{Int}
    m::Vector{Int}
    function HybridModel(models::Vector{<:AbstractModel}, 
            model_inds::Vector{<:AbstractVector},
            times::AbstractVector)

        # Check that model_inds have no overlap and no gaps
        all_inds = sort(Vector(union(model_inds...)))
        N = all_inds[end]
        @assert sum(length.(model_inds)) == N  # no overlap
        @assert all_inds == 1:N                # no gaps
        
        model_ids = zeros(Int, N)
        n = zeros(Int, N)
        m = zeros(Int, N)
        for (i,inds) in enumerate(model_inds)
            for k in inds
                model_ids[k] = i
                n[k] = state_dim(models[i])
                m[k] = control_dim(models[i])
            end
        end

        # Check for consistent transitions
        for k = 1:N-1
            model1 = models[model_ids[k]]
            model2 = models[model_ids[k+1]]
            n1 = next_state_dim(model1)    # output of dynamics function
            n2 = state_dim(model2)         # input to next dynamics function
            @assert n1 == n2 "Model dimension mismatch at timestep $k. Expected $n1, got $n2."
        end

        new(models, model_ids, times, n, m)
    end
end

get_times(model::HybridModel) = model.times
state_dim(model::HybridModel, k::Int) = model.n[k]
control_dim(model::HybridModel, k::Int) = model.m[k]

function dynamics(model::HybridModel, x, u, t)
    k = get_k(model, t)
    model_ind = model.model_inds[k]
    dynamics(model.models[model_ind], x, u, t)
end

function discrete_dynamics(::Type{Q}, model::HybridModel, x, u, t, dt) where Q
    k = get_k(model, t)
    model_ind = model.model_inds[k]
    discrete_dynamics(Q, model.models[model_ind], x, u, t, dt)
end


struct InitialControl{L,m,m0} <: AbstractHybridModel
    model::L
    initial::Function
    uinds0::SVector{m0,Int}
    uinds::SVector{m,Int}
end

state_dim(model::InitialControl) = state_dim(model.model)
control_dim(model::InitialControl{<:Any,m}, k::Int) where m = control_dim(model.model) + m * (k == 1)

function discrete_dynamics(::Type{Q}, model::InitialControl{<:Any,m}, x, u, t, dt) where {Q,m}
    # Call original dynamics
    u0 = u[model.uinds0]
    x2 = discrete_dynamics(Q, model.model, x, u0, t, dt)

    u_new = u[model.uinds]
    model.initial(model.model, x2, x, u0, u_new)
    # return [x2; u_new]
end
