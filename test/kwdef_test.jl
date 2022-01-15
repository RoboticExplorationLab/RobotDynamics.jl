using RobotDynamics
using ForwardDiff
using FiniteDiff
using Test
const RD = RobotDynamics

"""
My Docs
"""
RD.@autodiff Base.@kwdef struct MyKwModel <: RD.AbstractFunction
    a::Float64 = 1.0
    b::Float64 = 1.0
    int::Int = 10
end
RD.state_dim(::MyKwModel) = 4
RD.control_dim(::MyKwModel) = 2
RD.output_dim(::MyKwModel) = 4

function RD.evaluate(model::MyKwModel, x, u)
    return model.a * x + [u;u * model.int] * model.b
end

model = MyKwModel()
@test model.a == 1
@test model.int == 10
@test hasfield(MyKwModel, :cfg)
model = MyKwModel(a = 10, int=4)
@test model.a == 10
@test model.int == 4

x,u = rand(model)
y = zeros(length(x))
J = zeros(4,6)
z = RD.KnotPoint(x, u, 0., 0.1)
RD.evaluate(model, x, u)
RD.jacobian!(RD.StaticReturn(), RD.ForwardAD(), model, J, y, z)
A = ForwardDiff.jacobian(x->RD.evaluate(model, x, u), x)
B = ForwardDiff.jacobian(u->RD.evaluate(model, x, u), u)
@test J ≈ [A B]
