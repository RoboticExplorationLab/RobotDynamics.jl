# RobotDynamics.jl

```@meta
CurrentModule = RobotDynamics
```

Welcome to `RobotDynamics.jl`! This package is dedicated to providing a convenient interface
for defining the dynamics of forced dynamical systems, such as robots.

This package also provides many efficient methods for evaluating dynamics, their Jacobians,
and their discrete-time versions for use in optimization packages such as
[TrajectoryOptimization.jl](https://github.com/RoboticExplorationLab/TrajectoryOptimization.jl).


## Getting Started
```@example
a = 1
b = 2
a + b
```

### Defining a new continuous dynamics model
It is pretty straightforward to define a new dynamics model. We start by creating 
new struct that inherits from [`ContinuousDynamics`](@ref). You can store any model parameters in the model struct:

```@example quickstart; continued = true
using RobotDynamics
struct MyModel <: RobotDynamics.ContinuousDynamics
    mass::Float64
    spring_stiffness::Float64
end
```

We now need to specify the dimensions of our state and control vectors:

```@example quickstart; continued = true
RobotDynamics.state_dim(::MyModel) = 2
RobotDynamics.control_dim(::MyModel) = 1
```

!!! tip
    For best performance, these output of these functions should be static 
    with respect to the model type. For example, storing the state and control
    dimension as variables in the model and returning them can degrade  
    performance.

Now we're ready to define our dynamics functions. We can specify either in-place 
or out-of-place dynamics functions signatures. We start with the in-place 
definition:

```@example quickstart; continued = true
function RobotDynamics.dynamics!(model::MyModel, xdot, x, u)
    # x is the state vector
    # u is the control vector
    p = x[1]  # position 
    v = x[2]  # velocity
    pdot = v
    F = -model.spring_stiffness * p + u[1]  # force
    vdot = F / model.mass
    xdot[1] = pdot
    xdot[2] = vdot
    nothing
end
```

!!! warning
    Be sure to exactly match this function signature. Following standard Julia
    conventions, the in-place version uses a `!`, while the out-of-place method 
    below does not.

For small systems like this, it is often efficient to use out-of-place methods 
with [StaticArrays](https://github.com/JuliaArrays/StaticArrays.jl). Our signature
looks almost identical, except that it does not have a `!` at the end, and does 
not have `xdot` as a argument:

```@example quickstart; continued = true
using StaticArrays
function RobotDynamics.dynamics(model::MyModel, x, u)
    p = x[1]  # position 
    v = x[2]  # velocity
    pdot = v
    F = -model.spring_stiffness * p + u[1]  # force
    vdot = F / model.mass
    return SA[pdot, vdot]  # shortcut to create an SVector
end
```

### Calling the dynamics
Now that we've defined our dynamics, let's see how we can call it. This package 
provides the [`KnotPoint`](@ref) type that stores the state and control vector 
together with information about the time. Let's start by creating our model 
and some `KnotPoint` types:

```@example quickstart; continued = true 
mass = 1.0
stiffness = 0.5
model = MyModel(mass, stiffness)
n = RobotDynamics.state_dim(model)
m = RobotDynamics.control_dim(model)

# Create a KnotPoint using SVectors 
# (should use this when calling out-of-place methods)
xs = @SVector randn(n)
us = @SVector randn(m)
t = 0.0   # current time
dt = NaN  # time step, not needed for continuous models
zs = KnotPoint(xs, us, t, dt)

# Create a KnotPoint using normal vectors
# (using 's' for static, 'd' for dynamic)
xd = Vector(xs)
ud = Vector(us)
zd = KnotPoint{n,m}(xd, ud, t, dt)
```

Now that we've defined some inputs, let's evaluate our continuous dynamics. We 
can use any of the methods below:
```@example quickstart; continued = true 
# Out-of-place methods
RobotDynamics.dynamics(model, zs)
RobotDynamics.dynamics(model, xs, us)
RobotDynamics.dynamics(model, xs, us, t)

# In-place methods
xdot = zeros(n)
RobotDynamics.dynamics!(model, xdot, zd)
RobotDynamics.dynamics!(model, xdot, xd, ud)
RobotDynamics.dynamics!(model, xdot, xd, ud, t)
xd
```

### Querying the dynamics Jacobian 
Writing down an analytical dynamics Jacobian can often be time-consuming and 
error-prone. RobotDynamics allows the user to specify, at run time, the method 
to be used to evaluate the dynamics Jacobians, as long as they are defined on 
your type. To automatically define methods that use 
[ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) or
[FiniteDiff.jl](https://github.com/JuliaDiff/FiniteDiff.jl), you can use the
[`@autodiff`](@ref) macro provided by this package. This modifies your model,
adding new fields for the caches used by these methods, as well as the 
methods to evaluate them. For our example, all we have to do is add the macro 
before our model struct definition:

```@example autodiff; continued = true
using RobotDynamics     # hide
using StaticArrays      # hide
# NOTE: must include these packages when using `@autodiff`
using ForwardDiff
using FiniteDiff
RobotDynamics.@autodiff struct MyModel <: RobotDynamics.ContinuousDynamics
    mass::Float64
    spring_stiffness::Float64
end
RobotDynamics.state_dim(::MyModel) = 2    # hide
RobotDynamics.control_dim(::MyModel) = 1  # hide

function RobotDynamics.dynamics!(model::MyModel, xdot, x, u)  # hide
    # x is the state vector                                   # hide
    # u is the control vector                                 # hide
    p = x[1]                                                  # hide
    v = x[2]                                                  # hide
    pdot = v                                                  # hide
    F = -model.spring_stiffness * p + u[1]                    # hide
    vdot = F / model.mass                                     # hide
    xdot[1] = pdot                                            # hide
    xdot[2] = vdot                                            # hide
    nothing                                                   # hide
end                                                           # hide

function RobotDynamics.dynamics(model::MyModel, x, u)         # hide
    p = x[1]                                                  # hide
    v = x[2]                                                  # hide
    pdot = v                                                  # hide
    F = -model.spring_stiffness * p + u[1]                    # hide
    vdot = F / model.mass                                     # hide
    return SA[pdot, vdot]                                     # hide
end                                                           # hide
```

!!! warning
    Adding the `@autodiff` method before a type will require you to restart any 
    active Julia kernels you have running, since it changes the type definition
    by adding a type parameter and a couple extra fields related to the 
    caches for ForwardDiff and FiniteDiff. 

After which we can use the following generic methods to evaluate our dynamics
Jacobians:
```@example autodiff; continued = false 
mass = 1.0                          # hide
stiffness = 0.5                     # hide
model = MyModel(mass, stiffness)    # hide
n, m = RobotDynamics.dims(model)    # hide
xs = @SVector randn(n)              # hide
us = @SVector randn(m)              # hide
t = 0.0                             # hide
h = 0.1                             # hide
zs = KnotPoint(xs, us, t, h)        # hide
xd = Vector(xs)                     # hide
ud = Vector(us)                     # hide
zd = KnotPoint{n,m}(xd, ud, t, h)   # hide
xdot = zeros(n)                     # hide

J = zeros(n, n + m)
sig = RobotDynamics.StaticReturn()  # out-of-place signature
diff = RobotDynamics.ForwardAD()    # use forward AD differentiation
RobotDynamics.jacobian!(sig, diff, model, J, xdot, zs)

sig = RobotDynamics.InPlace()            # out-of-place signature
diff = RobotDynamics.FiniteDifference()  # use finite differences 
RobotDynamics.jacobian!(sig, diff, model, J, xdot, zd)
```

These methods should always be called using a `KnotPoint` input, since it's more
efficient to differentiation the dynamics treating the state and control as a 
single vector input into the dynamics, which is why the `KnotPoint` type stores
them as a concatenated vector.


### Discretizing our dynamics
We often need to discretize our continuous dynamics, either to simulate it or 
feed it into optimization frameworks. The [`DiscretizedModel`](@ref) type 
discretizes a [`ContinuousDynamics`](@ref) model, turning it into a 
[`DiscreteDynamics`](@ref) model by applying a [`QuadratureRule`](@ref).
To discretize our system using, e.g. Runge-Kutta 4, we create a new 
`DiscretizedDynamics` model:

```@example quickstart; continued = true
rk4 = RobotDynamics.RK4(n, m)  # create the integrator
model_discrete = RobotDynamics.DiscretizedDynamics(model, rk4)
```

We can query our discrete dynamics using very similar methods to those we 
used before, except now we have to specify the time step.

```@example quickstart; continued = false 
# Specify a time step of 0.1 seconds
zs.dt = 0.1
zd.dt = 0.1

# Out-of-place methods
RobotDynamics.discrete_dynamics(model_discrete, zs)
RobotDynamics.discrete_dynamics(model_discrete, xs, us, t, dt)

# In-place methods
xn = zeros(n)
RobotDynamics.discrete_dynamics!(model_discrete, xn, zd)
RobotDynamics.discrete_dynamics!(model_discrete, xn, xd, ud, t, dt)
```

The Jacobians are called using the same methods used for the continuous dynamics.