```@meta
CurrentModule = RobotDynamics
```

# Finite Differencing 
RobotDynamics allows the Jacobians to be calculated using finite 
differencing via FiniteDiff.jl
instead of forward-mode automatic differentiation via ForwardDiff.jl.
ForwardDiff tends to be faster, and obviously provides more accurate 
derivatives, but finite differencing will always work, even through
lookup tables, calling external code, or other functions that may be 
difficult to diff through using ForwardDiff.

To use finite differencing, set the `diffmethod` trait to `RobotDynamics.FiniteDifference()`. 
For example, if we want to use finite differencing
for the cartpole we can use the following line of code:
```julia
RobotDynamics.diffmethod(::Cartpole) = RobotDynamics.FiniteDifference()
```

The interface is exactly the same:
```julia
model = Cartpole()
z = KnotPoint(rand(model)..., 0.1)
F = RobotDynamics.DynamicsJacobian(model)
jacobian!(F, model, z)
```

To acheive zero allocation with FiniteDiff.jl, we need to pass in a `JacobianCache`. 
We can define this directly from the model and pass it to the Jacobian function:
```julia
cache = FiniteDiff.JacobianCache(model)
discrete_jacobian!(RK4, F, model, z, cache)
```

The difference method, datatype, and other options can be modified via the 
`JacobianCache` constructor:
```julia
cache = FiniteDiff.JacobianCache(model, Val(:central), Float64)
```