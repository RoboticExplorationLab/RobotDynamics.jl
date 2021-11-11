"""
    @autodef struct_def

Sets up methods to automatically evaluate the Jacobian of the function. By default, it defines
these methods for a new type `NewFun`:

    jacobian!(::InPlace, ::ForwardAD, fun::NewFun, J, y, z)
    jacobian!(::InPlace, ::FiniteDifference, fun::NewFun, J, y, z)
    jacobian!(::StaticReturn, ::ForwardAD, fun::NewFun, J, y, z)
    jacobian!(::StaticReturn, ::FiniteDifference, fun::NewFun, J, y, z)

These methods are all optimized to be non-allocating.

The method will modify the type definition, adding fields and type parameters to set them 
up for efficient evaluation of the jacobians. The changes are usually transparent to the user.

If an inner constructor is provided, the signature will be unchanged, but will be modified
to initialize the new fields and provide the new type parameters.

# Examples

```julia
@autodef struct MyFun <: RobotDynamics.AbstractFunction end
```

will define
```julia
struct MyFun{CH} <: RobotDynamics.AbstractFunction
    cfg::ForwardDiff.JacobianConfig{Nothing, Float64, CH, Tuple{Vector{ForwardDiff.Dual{Nothing, Float64, CH}}, Vector{(ForwardDiff).Dual{Nothing, Float64, CH}}}}
    cache::FiniteDiff.JacobianCache{Vector{Float64}, Vector{Float64}, Vector{Float64}, UnitRange{Int64}, Nothing, Val{:forward}(), Float64}
    function MyFun()
        model = new{0}()
        _n = input_dim(model)
        _m = output_dim(model)
        cfg = ForwardDiff.JacobianConfig(nothing, zeros(Float64, _m), zeros(Float64, _n))
        model = new{length(cfg).seeds}(cfg)
        _n = input_dim(model)
        _m = output_dim(model)
        cache = FiniteDiff.JacobianCache(zeros(Float64, _n), zeros(Float64, _m))
        model = new{length(cfg).seeds}(cfg, cache)
    end
end

function RobotDynamics.jacobian!(::InPlace, ::ForwardAD, fun::MyFun, J, y, z)
    f_aug!(y, z_) = begin
        RobotDynamics.evaluate!(fun, y, RobotDynamics.setinputs!(z, z_))
        ForwardDiff.jacobian!(J, f_aug!, y, z, fun.cfg)
    end
end
...  # other Jacobian methods
```

"""
macro autodiff(type::Expr)
    _autodiff(__module__, type, [InPlace(), StaticReturn()], [ForwardAD(), FiniteDifference()])
end

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
    _autodiff(__module__, type, sigs, methods)
end

function _autodiff(mod, struct_expr::Expr, sigs::Vector{FunctionSignature}, methods::Vector{DiffMethod})
    jac_defs = Expr[]
    
    # Get signature for jacobian methods 
    type_expr = struct_expr.args[2].args[1]

    for method in methods
        for sig in sigs
            push!(jac_defs, gen_jacobian(sig, method, type_expr, mod))
        end
        struct_expr = modify_struct_def(method, struct_expr, mod)
    end
    return quote
        $struct_expr
        $(Expr(:block, jac_defs...))
    end
end

function get_call_signature(sig, diff, type_expr, mod)
    # Get the name of the type
    local type_name
    if type_expr isa Expr
        if type_expr.head == :curly
            type_name = type_expr.args[1]
        else
            @show type_expr
            error("Got an unexpected type expression. Can't generate the Jacobian.")
        end
    elseif type_expr isa Symbol
        type_name = type_expr
    end

    # Convert type to a GlobalRef
    (type_name isa Symbol) && (type_name = GlobalRef(mod, type_name))

    # Create the call signature
    Expr(:call, GlobalRef(@__MODULE__, :jacobian!), :(::$(typeof(sig))), :(::$(typeof(diff))), :(fun::$type_name), :J, :y, :z)
end

"""
Takes an inner constructor of the form:
```
function MyType(args...)
    ...
    new(args...)
end
```
and adds type params:
```
function MyType{params...}(args...) where {params...}
    ...
    new{params...}(args...)
end
```

"""
function add_params_to_innercon(fun, params)
    @assert fun.head == :function
    sig = fun.args[1]
    newsig = :($(copy(sig)) where {$(params...)})
    typename = newsig.args[1].args[1]
    newsig.args[1].args[1] = :($typename{$(params...)})

    newfun = copy(fun)
    newfun.args[1] = newsig
    newcall = newfun.args[2].args[end]
    newcall.args[1] = :(new{$(params...)})
    newfun
end

function get_struct_parent(struct_expr::Expr)
    @assert struct_expr.head == :struct
    typedef = struct_expr.args[2]
    if typedef.head != :(<:)
        error("Struct has no parent!")
    else
        return typedef.args[2]
    end
end

function add_field_to_struct(struct_expr0::Expr, newfield::Expr, init_field, pname::Union{Symbol,Nothing}, init_param, mod)

    struct_expr = copy(struct_expr0)
    @assert struct_expr.head == :struct
    typedef = struct_expr.args[2]
    body = struct_expr.args[3]

    # Check for valid sub-typing
    parent_name = get_struct_parent(struct_expr)
    parent = mod.eval(parent_name)
    if !(parent <: RobotDynamics.AbstractFunction)
        error("Type must be a sub-type of RobotDynamics.AbstractFunction")
    end
    type_param = Symbol(inputtype(parent))

    # Resolve the parent name in the original scope
    if parent_name isa Symbol
        typedef.args[2] = GlobalRef(mod, parent_name)
    end

    # Add type parameter 
    has_params = false
    if !isnothing(pname)
        if typedef.args[1] isa Expr && typedef.args[1].head == :curly 
            has_params = true
            typeparams = typedef.args[1].args

            # Check if the parameter name alraedy exists
            for param in typeparams
                if (param isa Symbol && param == pname) || (param isa GlobalRef && param.name == pname)
                    error("Cannot add parameter, the type parameter $pname is already being used.")
                end
            end

            # Add to type params
            push!(typeparams, pname)
        elseif typedef.args[1] isa Symbol  # add new param
            typename = typedef.args[1]
            typedef.args[1] = :($typename{$pname})
        end
    end

    # Add field
    @assert newfield isa Expr && newfield.head == :(::)
    name = newfield.args[1]

    # Check if the field name already exists
    for field in body.args
        if field isa Expr && field.head == :(::) && field.args[1] == name 
            error("Cannot add field, the field $name already exists")
        end
    end

    # Add the field to the end of the struct
    push!(body.args, newfield)

    # Modify the constructor
    constructor_found = false
    for field in body.args
        if field isa Expr && field.head == :function
            constructor_found |= add_config_to_constructor(field, type_param, name, init_field, init_param)
        end
    end

    # Add new constructor, if needed
    if !constructor_found
        inner_con = new_default_constructor(struct_expr0)
        push!(body.args, inner_con)
        add_config_to_constructor(inner_con, type_param, name, init_field, init_param)
    end
    return struct_expr
end

function new_default_constructor(struct_expr::Expr)
    @assert struct_expr.head == :struct
    struct_expr = copy(struct_expr)
    typedef = struct_expr.args[2]
    body = struct_expr.args[3]

    # Get the type name
    if typedef.head == :(<:)
        typename = typedef.args[1]
    else
        typename = typedef
    end

    # Get parameters
    params = [] 
    if typename isa Expr && typename.head == :curly
        params = typename.args[2:end]
    end

    # Get the field names
    args = map(filter(body.args) do arg
        (arg isa Expr && arg.head == :(::)) || arg isa Symbol
    end) do arg
        arg isa Expr ? arg.args[1] : arg
    end

    # Call to "new"
    inner_con = quote
        function $typename($(args...))
            new($(args...))
        end
    end
    inner_con = inner_con.args[2]
    @assert inner_con.head == :function

    # Add type params
    if !isempty(params)
        # Add "where" clause
        call = inner_con.args[1]
        params = [:T,:T2]
        inner_con.args[1] = Expr(:where, call, params...)

        # Add to "new"
        newcall = inner_con.args[2].args[end]
        newcall.args[1] = :(new{$(params...)})
    end

    return inner_con 
end

function add_config_to_constructor(con, type_param, fieldname, init_field, init_param)
    @assert con.head == :function 
    is_constructor = false
    constructor_body = con.args[2]
    for (j, line) in enumerate(reverse(constructor_body.args))
        i = length(constructor_body.args) - j + 1  # line number

        isnewcall(call) = call.args[1] == :new || (call.args[1] isa Expr && call.args[1].head == :curly && call.args[1].args[1] == :new)
        hasnewcall = line isa Expr && line.head == :call && isnewcall(line)
        hasnewassign = line isa Expr && line.head == :(=) && line.args[2] isa Expr && line.args[2].head == :call && isnewcall(line.args[2]) 
        if hasnewcall 
            callfun = line
        elseif hasnewassign
            callfun = line.args[2]
        else
            continue
        end

        newparam = init_param(fieldname)
        add_newparam = !isnothing(newparam)

        # Add new param, so that the "new" call looks like
        # new{...,new_param}(...)
        if add_newparam
            @assert callfun.head == :call
            hasparams = callfun.args[1] isa Expr
            new_param0 = :(0)
            if hasparams  # append type param
                @assert callfun.args[1].head == :curly
                push!(callfun.args[1].args, new_param0)
            else  # add new type param
                @assert callfun.args[1] == :new
                callfun.args[1] = :(new{$new_param0}) 
            end
        end

        # Append assignment to make the line look like
        # model = new{..., new_param}(...)
        outname = :__model__
        if hasnewcall
            line = :($outname = $line)
            constructor_body.args[i] = line
        else
            # if a model is output, create an alias with the given name
            @assert line.head == :(=)
            if outname != line.args[1]
                insert!(constructor_body.args, i+1, :($outname = $(line.args[1])))
            end
        end

        # add the new argument to "new"
        callfun_new = copy(callfun)
        if add_newparam 
            callfun_new.args[1].args[end] = newparam 
        end
        push!(callfun_new.args, fieldname)  # add to the arguments to "new"

        # add lines after the call to "new" that initialize the new variables 
        init_expr = init_field(outname, fieldname, type_param, callfun_new)
        for arg in reverse(init_expr.args)
            insert!(constructor_body.args, i+1, arg)
        end
        is_constructor = true
        break
    end
    return is_constructor
end


#########################
# ForwardDiff
#########################

function gen_jacobian(sig::StaticReturn, diff::ForwardAD, type_expr, mod)
    callsig = get_call_signature(sig, diff, type_expr, mod)

    eval = GlobalRef(@__MODULE__, :evaluate)
    fun_body = quote
        f(_z) = $eval(fun, getstate(z, _z), getcontrol(z, _z), getparams(z))
        J .= ForwardDiff.jacobian(f, getdata(z))
        return nothing
    end
    Expr(:function, callsig, fun_body)
end

function gen_jacobian(sig::InPlace, diff::ForwardAD, type_expr, mod)
    callsig = get_call_signature(sig, diff, type_expr, mod)

    eval = GlobalRef(@__MODULE__, :(evaluate!))
    fun_body = quote
        f!(_y,_z) = $eval(fun, _y, getstate(z, _z), getcontrol(z, _z), getparams(z))
        ForwardDiff.jacobian!(J, f!, y, getdata(z), fun.cfg)
        return nothing
    end
    Expr(:function, callsig, fun_body)
end

function init_cfg(outname, fieldname, type_param, callfun)
    quote
        _n = RobotDynamics.state_dim($outname) + RobotDynamics.control_dim($outname)
        _m = RobotDynamics.output_dim($outname)
        $fieldname = ForwardDiff.JacobianConfig(nothing, zeros($type_param, _m), zeros($type_param, _n))
        $outname = $callfun
    end
end

function init_cfg_param(fieldname)
    :(length($fieldname.seeds))
end

function modify_struct_def(::ForwardAD, struct_expr::Expr, mod)
    pname = :JCH
    parent_name = get_struct_parent(struct_expr)
    parent = mod.eval(parent_name)
    type_param = Symbol(inputtype(parent))
    newfield = :(cfg::ForwardDiff.JacobianConfig{Nothing, $type_param, $pname, Tuple{Vector{ForwardDiff.Dual{Nothing, $type_param, $pname}}, Vector{ForwardDiff.Dual{Nothing, $type_param, $pname}}}})

    struct_expr = copy(struct_expr)
    @assert struct_expr.head == :struct

    return add_field_to_struct(struct_expr, newfield, init_cfg, pname, init_cfg_param, mod)
end

#########################
# FiniteDiff 
#########################

function gen_jacobian(sig::StaticReturn, diff::FiniteDifference, type_expr, mod)
    callsig = get_call_signature(sig, diff, type_expr, mod)

    eval = GlobalRef(@__MODULE__, :evaluate)
    fun_body = quote
        f!(_y,_z) = _y .= $eval(fun, getstate(z, _z), getcontrol(z, _z), getparams(z))
        FiniteDiff.finite_difference_jacobian!(J, f!, getdata(z), fun.cache)
        return nothing
    end
    Expr(:function, callsig, fun_body)
end

function gen_jacobian(sig::InPlace, diff::FiniteDifference, type_expr, mod)
    callsig = get_call_signature(sig, diff, type_expr, mod)

    eval = GlobalRef(@__MODULE__, :evaluate!)
    fun_body = quote
        f!(_y,_z) = $eval(fun, _y, getstate(z, _z), getcontrol(z, _z), getparams(z))
        FiniteDiff.finite_difference_jacobian!(J, f!, getdata(z), fun.cache)
        return nothing
    end
    Expr(:function, callsig, fun_body)
end

function init_cache(outname, fieldname, type_param, callfun)
    quote
        _n = RobotDynamics.state_dim($outname) + RobotDynamics.control_dim($outname)
        _m = RobotDynamics.output_dim($outname)
        $fieldname = FiniteDiff.JacobianCache(zeros($type_param, _n), zeros($type_param, _m))
        $outname = $callfun
    end
end

init_cache_param(fieldname) = nothing

function modify_struct_def(::FiniteDifference, struct_expr::Expr, mod)
    pname = nothing 
    parent_name = get_struct_parent(struct_expr)
    parent = mod.eval(parent_name)
    type_param = Symbol(inputtype(parent))
    newfield = :(cache::FiniteDiff.JacobianCache{Vector{$type_param}, Vector{$type_param}, Vector{$type_param}, UnitRange{Int64}, Nothing, Val{:forward}(), $type_param})

    struct_expr = copy(struct_expr)
    @assert struct_expr.head == :struct

    return add_field_to_struct(struct_expr, newfield, init_cache, pname, init_cache_param, mod)
end