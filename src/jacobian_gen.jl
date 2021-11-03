macro autodiff(type::Expr, args...)
    sigs = FunctionSignature[]
    methods = DiffMethod[]
    for arg in args
        if arg isa Symbol
            earg = eval(arg)
            if earg <: FunctionSignature
                push!(sigs, earg())
            elseif earg <: DiffMethod
                push!(methods, earg())
            end
        end
    end
    _autodiff(type, sigs, methods)
end

function _autodiff(struct_expr::Expr, sigs::Vector{FunctionSignature}, methods::Vector{DiffMethod})
    jac_defs = Expr[]
    type_expr = copy(struct_expr.args[2].args[1])  # get the type definition (with type params)
    for method in methods
        for sig in sigs
            push!(jac_defs, gen_jacobian(sig, method, type_expr))
        end
        struct_expr = modify_struct_def(method, struct_expr)
    end
    return quote
        $struct_expr
        $(Expr(:block, jac_defs...))
    end
end

#########################
# ForwardDiff
#########################

function gen_jacobian(sig::StaticReturn, diff::ForwardAD, type_expr::Expr)
    @assert type_expr.head == :curly 

    # Convert type to a GlobalRef
    mod = @__MODULE__
    (type_expr.args[1] isa Symbol) && (type_expr.args[1] = GlobalRef(mod, type_expr.args[1]))

    # Create call signature
    nparams = length(type_expr.args) - 1
    type_params = [type_expr.args[i] for i = 2:nparams + 1]
    callsig = Expr(:call, GlobalRef(mod, :jacobian!), :(::$(typeof(sig))), :(::$(typeof(diff))), :(fun::$type_expr), :J, :y, :z)
    wheresig = Expr(:where, callsig, type_params...)
    fun_body = quote
        n = state_dim(typeof(fun))
        m = control_dim(typeof(fun))
        ix = SVector{n}(1:n)
        iu = SVector{m}(n+1:n+m)
        f_aug(z) = evaluate(fun, z[ix], z[iu]) 
        J .= ForwardDiff.jacobian(f_aug, z)
        return nothing
    end
    Expr(:function, wheresig, fun_body)
end

function gen_jacobian(sig::InPlace, diff::ForwardAD, type_expr::Expr)
    @assert type_expr.head == :curly 

    # Convert type to a GlobalRef
    mod = @__MODULE__
    (type_expr.args[1] isa Symbol) && (type_expr.args[1] = GlobalRef(mod, type_expr.args[1]))

    # Create call signature
    nparams = length(type_expr.args) - 1
    type_params = [type_expr.args[i] for i = 2:nparams + 1]
    callsig = Expr(:call, GlobalRef(mod, :jacobian!), :(::$(typeof(sig))), :(::$(typeof(diff))), :(fun::$type_expr), :J, :y, :z)
    wheresig = Expr(:where, callsig, type_params...)
    fun_body = quote
        n = state_dim(fun)
        m = control_dim(fun)
        f_aug!(y, z) = evaluate!(fun, y, view(z, 1:n), view(z, n+1:n+m))
        ForwardDiff.jacobian!(J, f_aug!, y, z, fun.cfg)
        return nothing
    end
    Expr(:function, wheresig, fun_body)
end

function modify_struct_def(::ForwardAD, struct_expr::Expr)
    struct_expr = copy(struct_expr)
    @assert struct_expr.head == :struct

    # Add chunk size param
    typedef = struct_expr.args[2]
    typeparams = typedef.args[1].args
    typeparams0 = copy(typeparams)
    chunk_param = :CH
    for param in typeparams
        if (param isa Symbol && param == chunk_param) || (param isa GlobalRef && param.name == chunk_param)
            error("Cannot add JacobianConfig field, the type parameter $chunk_param is already being used.")
        end
    end
    push!(typeparams, :CH)    # add param to type

    # Add extra field to type for JacobianConfig
    body = struct_expr.args[3] 
    config_fieldname = :cfg
    for field in body.args
        if field isa Expr && field.head == :(::) && field.args[1] == config_fieldname
            error("Cannot add JacobianConfig field, the field $config_fieldname already exists")
        end
    end
    type_param = :Float64
    config_field = :(cfg::(ForwardDiff).JacobianConfig{Nothing, $type_param, $chunk_param, Tuple{Vector{(ForwardDiff).Dual{Nothing, $type_param, $chunk_param}}, Vector{(ForwardDiff).Dual{Nothing, $type_param, $chunk_param}}}})
    insert!(body.args, 1, config_field)

    # Check for inner constructors
    constructor_found = false
    for field in body.args
        if field isa Expr && field.head == :function
            constructor_found |= add_ad_config_to_constructor(field, config_fieldname, type_param)
        end
    end
    return struct_expr
end

function add_ad_config_to_constructor(con::Expr, config_fieldname::Symbol, type_param::Symbol)
    @assert con.head == :function

    # Find call to "new"
    is_constructor = false
    constructor_body = con.args[2]
    for (i,line) in enumerate(constructor_body.args)
        if line isa Expr && line.head == :call
            insert!(line.args, 2, config_fieldname)  # add to the arguments to "new"
            callfun = line.args[1]
            if callfun isa Expr && callfun.head == :curly && callfun.args[1] == :new
                is_constructor = true
                init_cfg = :(cfg = ForwardDiff.JacobianConfig(nothing, zeros($type_param, n), zeros($type_param, n+m)))
                insert!(constructor_body.args, i, init_cfg)  # add a line that initializes the cfg before the call to "new"
                push!(callfun.args, :(length($(config_fieldname).seeds)))  # add type param to call to "new"
                break
            end
        end
    end
    return is_constructor 
end

#########################
# FiniteDiff 
#########################

function gen_jacobian(sig::StaticReturn, diff::FiniteDifference, type_expr::Expr)
    @assert type_expr.head == :curly 

    # Convert type to a GlobalRef
    mod = @__MODULE__
    (type_expr.args[1] isa Symbol) && (type_expr.args[1] = GlobalRef(mod, type_expr.args[1]))

    # Create call signature
    nparams = length(type_expr.args) - 1
    type_params = [type_expr.args[i] for i = 2:nparams + 1]
    callsig = Expr(:call, GlobalRef(mod, :jacobian!), :(::$(typeof(sig))), :(::$(typeof(diff))), :(fun::$type_expr), :J, :y, :z)
    wheresig = Expr(:where, callsig, type_params...)
    fun_body = quote
        n = state_dim(fun) 
        m = control_dim(fun)
        f_aug!(y, z) = y .= evaluate(fun, view(z, 1:n), view(z, n+1:n+m))
        FiniteDiff.finite_difference_jacobian!(J, f_aug!, z, fun.cache)
        return nothing
    end
    Expr(:function, wheresig, fun_body)
end

function gen_jacobian(sig::InPlace, diff::FiniteDifference, type_expr::Expr)
    @assert type_expr.head == :curly 

    # Convert type to a GlobalRef
    mod = @__MODULE__
    (type_expr.args[1] isa Symbol) && (type_expr.args[1] = GlobalRef(mod, type_expr.args[1]))

    # Create call signature
    nparams = length(type_expr.args) - 1
    type_params = [type_expr.args[i] for i = 2:nparams + 1]
    callsig = Expr(:call, GlobalRef(mod, :jacobian!), :(::$(typeof(sig))), :(::$(typeof(diff))), :(fun::$type_expr), :J, :y, :z)
    wheresig = Expr(:where, callsig, type_params...)
    fun_body = quote
        n = state_dim(fun) 
        m = control_dim(fun)
        f_aug!(y, z) = evaluate!(fun, y, view(z, 1:n), view(z, n+1:n+m))
        FiniteDiff.finite_difference_jacobian!(J, f_aug!, z, fun.cache)
        return nothing
    end
    Expr(:function, wheresig, fun_body)
end

function modify_struct_def(::FiniteDifference, struct_expr::Expr)
    struct_expr = copy(struct_expr)
    @assert struct_expr.head == :struct

    # Add extra field to type for JacobianCache
    body = struct_expr.args[3]
    cache_fieldname = :cache
    for field in body.args
        if field isa Expr && field.head == :(::) && field.args[1] == cache_fieldname
            error("Cannot add JacobianCache field, the field $cache_fieldname already exists")
        end
    end
    type_param = :Float64
    cache_field = :(cache::FiniteDiff.JacobianCache{Vector{$type_param}, Vector{$type_param}, Vector{$type_param}, UnitRange{Int64}, Nothing, Val{:forward}(), $type_param})
    insert!(body.args, 1, cache_field)

    # Check for inner constructors
    constructor_found = false
    for field in body.args
        if field isa Expr && field.head == :function
            constructor_found |= add_fd_cache_to_constructor(field, cache_fieldname, type_param)
        end
    end
    return struct_expr
end

function add_fd_cache_to_constructor(con::Expr, cache_fieldname::Symbol, type_param::Symbol)
    @assert con.head == :function
    # Find call to "new"
    is_constructor = false
    constructor_body = con.args[2]
    for (i,line) in enumerate(constructor_body.args)
        if line isa Expr && line.head == :call
            insert!(line.args, 2, cache_fieldname)  # add to the arguments to "new"
            callfun = line.args[1]
            if callfun isa Expr && callfun.head == :curly && callfun.args[1] == :new
                is_constructor = true
                init_cache = :(cache = FiniteDiff.JacobianCache(zeros($type_param, n+m), zeros($type_param, n)))
                insert!(constructor_body.args, i, init_cache)
                break
            end
        end
    end
    return is_constructor 
end
