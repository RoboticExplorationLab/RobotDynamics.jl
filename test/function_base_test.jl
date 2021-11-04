using RobotDynamics

struct TestFun <: RobotDynamics.AbsractFunction end

function evaluate(::TestFun, x, u)

end
