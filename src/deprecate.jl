@deprecate state_diff_jacobian!(args...) errstate_jacobian!(args...)
@deprecate ∇²differential!(args...) ∇errstate_jacobian!(args...)

import Base.size
@deprecate size(fun::AbstractFunction) dims(fun)
