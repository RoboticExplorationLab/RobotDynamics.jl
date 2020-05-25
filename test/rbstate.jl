using Test
using RobotDynamics
using StaticArrays
using Rotations
using BenchmarkTools

r = @SVector rand(3)
q = rand(UnitQuaternion)
v = @SVector rand(3)
ω = @SVector rand(3)

x = RBState{Float64}(r, q, v, ω)
@test x[1:3] ≈ r ≈ position(x)
@test x[4:7] ≈ Rotations.params(q)
@test x[8:10] ≈ v ≈ linear_velocity(x)
@test x[11:13] ≈ ω ≈ angular_velocity(x)
@test length(x) == 13

x32 = RBState{Float32}(r, q, v, ω)
@test x32 isa RBState{Float32}
@test eltype(x32[1:3]) == Float32
@test position(x32) isa SVector{3,Float32}
@test orientation(x32) isa UnitQuaternion{Float32}

# Pass in other types of vectors
x = RBState{Float64}(Vector(r), q, v, ω)
@test position(x) isa SVector{3,Float64}
x = RBState{Float64}(Vector(r), q, Vector(v), Vector(ω))
@test position(x) isa SVector{3,Float64}
@test linear_velocity(x) isa SVector{3,Float64}
@test angular_velocity(x) isa SVector{3,Float64}

x = RBState{Float64}(view(r,:), q, view(Vector(v),1:3), Vector(ω))
@test position(x) isa SVector{3,Float64}

# Pass in other rotations
x2 = RBState{Float64}(r, RotMatrix(q), v, ω)
@test orientation(x2) \ orientation(x) ≈ one(UnitQuaternion)
@test orientation(x2) isa UnitQuaternion

# Let the constructor figure out data type
@test RBState(r, q, v, ω) isa RBState{Float64}
@test RBState(Float32.(r), UnitQuaternion{Float32}(q), Float32.(v), Float32.(ω)) isa RBState{Float32}
@test RBState(Float32.(r), q, Float32.(v), Float32.(ω)) isa RBState{Float64}
@test RBState(r, UnitQuaternion{Float32}(q), Float32.(v), Float32.(ω)) isa RBState{Float64}

# Pass in a vector for the quaternion
q_ = Rotations.params(q)
x = RBState(r, q_, v, ω)
@test orientation(x) ≈ q
x = RBState(r, 2q_, v, ω)
@test orientation(x) ≈ q  # should renormalize
@test RBState(Float32.(r), Float32.(q_), Float32.(v), Float32.(ω)) isa RBState{Float32}

# Pass in a vector for the entire state
x_ = [r; 2q_; v; ω]
x = RBState(x_)
@test position(x) ≈ r
@test orientation(x) ≈ q  # should renormalize
@test linear_velocity(x) ≈ v
@test angular_velocity(x) ≈ ω

@test RBState{Float32}(x_) isa RBState{Float32}
@test RBState(Vector(x_)) isa RBState{Float64}

# Test comparison (with double-cover)
x1 = RBState(r, q, v, ω)
x2 = RBState(r, -q, v, ω)
@test x1[4:7] ≈ -x2[4:7]
@test x1 ≈ x2
@test !(SVector(x1) ≈ SVector(x2))
