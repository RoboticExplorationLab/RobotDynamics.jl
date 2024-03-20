using Test
using RobotDynamics
using StaticArrays
using Rotations
using BenchmarkTools
using Random

Random.seed!(1)
r = @SVector rand(3)
q = rand(QuatRotation)
v = @SVector rand(3)
ω = @SVector rand(3)

x = RBState{Float64}(r, q, v, ω)
@test x[1:3] ≈ r ≈ position(x)
@test x[4:7] ≈ Rotations.params(q)
@test x[8:10] ≈ v ≈ RD.linear_velocity(x)
@test x[11:13] ≈ ω ≈ RD.angular_velocity(x)
@test length(x) == 13

x32 = RBState{Float32}(r, q, v, ω)
@test x32 isa RBState{Float32}
@test eltype(x32[1:3]) == Float32
@test RD.position(x32) isa SVector{3,Float32}
@test RD.orientation(x32) isa QuatRotation{Float32}

# Pass in other types of vectors
x = RBState{Float64}(Vector(r), q, v, ω)
@test RD.position(x) isa SVector{3,Float64}
x = RBState{Float64}(Vector(r), q, Vector(v), Vector(ω))
@test RD.position(x) isa SVector{3,Float64}
@test RD.linear_velocity(x) isa SVector{3,Float64}
@test RD.angular_velocity(x) isa SVector{3,Float64}

x = RBState{Float64}(view(r, :), q, view(Vector(v), 1:3), Vector(ω))
@test RD.position(x) isa SVector{3,Float64}

@test RD.RBState{Float32}(x) isa RBState{Float32}
@test RD.RBState{Float64}(x32) isa RBState{Float64}
@test RD.RBState(x) isa RBState{Float64}
@test RD.RBState(x32) isa RBState{Float32}

# Pass in other rotations
x2 = RD.RBState{Float64}(r, RotMatrix(q), v, ω)
@test RD.orientation(x2) \ RD.orientation(x) ≈ one(QuatRotation)
@test RD.orientation(x2) isa QuatRotation

# Let the constructor figure out data type
@test RBState(r, q, v, ω) isa RBState{Float64}
@test RBState(Float32.(r), QuatRotation{Float32}(q), Float32.(v), Float32.(ω)) isa RBState{Float32}
@test RBState(Float32.(r), q, Float32.(v), Float32.(ω)) isa RBState{Float64}
@test RBState(r, QuatRotation{Float32}(q), Float32.(v), Float32.(ω)) isa RBState{Float64}

# Convert from another rotation type
x = [r; Rotations.params(MRP(q)); v; ω]
@test RBState(MRP, x) ≈ RBState(r, q, v, ω)
x = [r; Rotations.params(RodriguesParam(q)); v; ω]
@test !(RBState(MRP, x) ≈ RBState(r, q, v, ω))
@test RBState(RodriguesParam, x) ≈ RBState(r, q, v, ω)


# Pass in a vector for the quaternion
q_ = Rotations.params(q)
x = RBState(r, q_, v, ω)
@test RD.orientation(x) ≈ q
x = RBState(r, 2q_, v, ω)
@which RBState(r, 2q_, v, ω)
@test Rotations.params(RD.orientation(x)) ≈ 2q_  # shouldnt renormalize
Rotations.params(RD.orientation(x))
@test RBState(Float32.(r), Float32.(q_), Float32.(v), Float32.(ω)) isa RBState{Float32}

# Pass in a vector for the entire state
using RobotDynamics: position, orientation, linear_velocity, angular_velocity
x_ = [r; 2q_; v; ω]
x = RBState(x_)
@test position(x) ≈ r
@test Rotations.params(orientation(x)) ≈ 2q_  # shouldnt renormalize
@test linear_velocity(x) ≈ v
@test angular_velocity(x) ≈ ω

@test RBState{Float32}(x_) isa RBState{Float32}
@test RBState(Vector(x_)) isa RBState{Float64}

# Test comparison (with double-cover)
q = rand(QuatRotation)
x1 = RBState(r, q, v, ω)
x2 = RBState(r, -q.q, v, ω)
@which RBState(r, -q.q, v, ω)
@test x1[4:7] ≈ -x2[4:7]
@test x1 ≈ x2
@test !(SVector(x1) ≈ SVector(x2))

# Test indexing
x = RBState(r, q, v, ω)
@test x[1:3] ≈ r
@test x[4:7] ≈ Rotations.params(q)
@test x[8:10] ≈ v
@test x[11:13] ≈ ω
@test x[4] ≈ q.w

@test getindex(x, 2) == r[2]
@test getindex(x, 8) == v[1]
@test getindex(x, 13) == ω[3]

# Renorm
x2 = RBState(r, 2 * q.q, v, ω)
@test norm(orientation(x2).q) ≈ 2
x = RobotDynamics.renorm(x2)
@test norm(orientation(x2).q) ≈ 2
@test norm(orientation(x).q) ≈ 1
@test isrotation(orientation(x))

# Test zero constructor
x0 = zero(RBState)
@test position(x0) ≈ zero(r)
@test orientation(x0) ≈ one(QuatRotation)
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

@test dx isa SVector{12}
@test x2 ⊕ dx ≈ x1

q1 = orientation(x1)
q2 = orientation(x2)
@test Rotations.params(RodriguesParam(q2) \ RodriguesParam(q1)) ≈ dx[4:6]

# Test randbetween
xmin = RBState(fill(-1, 3), one(QuatRotation), fill(-2, 3), fill(-3, 3))
xmax = RBState(fill(+1, 3), one(QuatRotation), fill(+2, 3), fill(+3, 3))
@test randbetween(xmin, xmax) isa RBState{Float64}
for k = 1:10
    local x = randbetween(xmin, xmax)
    @test all(position(xmin) .<= position(x) .<= position(xmax))
    @test all(linear_velocity(xmin) .<= linear_velocity(x) .<= linear_velocity(xmax))
    @test all(angular_velocity(xmin) .<= angular_velocity(x) .<= angular_velocity(xmax))
end