module Dynamics

using DocStringExtensions
using DifferentialRotations
using StaticArrays
using LinearAlgebra
using ForwardDiff

include("expansion.jl")
include("knotpoint.jl")
include("model.jl")
include("rigidbody.jl")
include("integration.jl")
include("trajectories.jl")

end # module
