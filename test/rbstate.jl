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

@test RBState{Float32}(x) isa RBState{Float32}
@test RBState{Float64}(x32) isa RBState{Float64}
@test RBState(x) isa RBState{Float64}
@test RBState(x32) isa RBState{Float32}

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
@test orientation(x) ≈ 2q  # shouldnt renormalize
@test RBState(Float32.(r), Float32.(q_), Float32.(v), Float32.(ω)) isa RBState{Float32}

# Pass in a vector for the entire state
x_ = [r; 2q_; v; ω]
x = RBState(x_)
@test position(x) ≈ r
@test orientation(x) ≈ 2q  # shouldnt renormalize
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

# Test indexing
@test x[1:3] ≈ r
@test x[4:7] ≈ Rotations.params(q)
@test x[8:10] ≈ v
@test x[11:13] ≈ ω
@test x[4] ≈ q.w

# Renorm
x2 = RBState(r,2*q,v,ω)
@test norm(orientation(x2)) ≈ 2
x = RobotDynamics.renorm(x2)
@test norm(orientation(x2)) ≈ 2
@test norm(orientation(x)) ≈ 1
@test isrotation(orientation(x))

# Test zero constructor
x0 = zero(RBState)
@test position(x0) ≈ zero(r)
@test orientation(x0) ≈ one(UnitQuaternion)
@test linear_velocity(x0) ≈ zero(v)
@test angular_velocity(x0) ≈ zero(v)
@test zero(RBState) isa RBState{Float64}
@test zero(RBState{Float32}) isa RBState{Float32}
x = rand(RBState)
@test zero(x) ≈ x0

# Test random generator
x = rand(RBState)
@test isrotation(orientation(x))
@test x isa RBState{Float64}
@test position(x) isa SVector{3,Float64}
x32 = rand(RBState{Float32})
@test isrotation(orientation(x))
@test x32 isa RBState{Float32}
@test position(x32) isa SVector{3,Float32}

# Addition and Subtraction
x1 = rand(RBState)
x2 = rand(RBState)
x = x1 + x2
@test position(x) ≈ position(x1) + position(x2)
@test orientation(x) ≈ orientation(x1) * orientation(x2)
@test linear_velocity(x) ≈ linear_velocity(x1) + linear_velocity(x2)
@test angular_velocity(x) ≈ angular_velocity(x1) + angular_velocity(x2)
x = x1 - x2
@test position(x) ≈ position(x1) - position(x2)
@test orientation(x) ≈ orientation(x2) \ orientation(x1)
@test linear_velocity(x) ≈ linear_velocity(x1) - linear_velocity(x2)
@test angular_velocity(x) ≈ angular_velocity(x1) - angular_velocity(x2)

dx = x1 ⊖ x2
@test dx[1:3] ≈ position(x1) - position(x2)
@test dx[4:6] ≈ orientation(x1) ⊖ orientation(x2)
@test dx[7:9] ≈ linear_velocity(x1) - linear_velocity(x2)
@test dx[10:12] ≈ angular_velocity(x1) - angular_velocity(x2)

q1 = orientation(x1)
q2 = orientation(x2)
@test Rotations.params(RodriguesParam(q2) \ RodriguesParam(q1)) ≈ dx[4:6]

# Test randbetween
xmin = RBState(fill(-1,3), one(UnitQuaternion), fill(-2,3), fill(-3,3))
xmax = RBState(fill(+1,3), one(UnitQuaternion), fill(+2,3), fill(+3,3))
@test randbetween(xmin, xmax) isa RBState{Float64}
for k = 1:10
    x = randbetween(xmin, xmax)
    @test all(position(xmin) .<= position(x) .<= position(xmax))
    @test all(linear_velocity(xmin) .<= linear_velocity(x) .<= linear_velocity(xmax))
    @test all(angular_velocity(xmin) .<= angular_velocity(x) .<= angular_velocity(xmax))
end
