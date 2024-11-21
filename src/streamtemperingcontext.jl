"""
Context that multiplies each log-prior by mod
used to test whether varwise_logpriors respects child-context.
"""
struct StreamTemperingContext{T,Ctx} <: DynamicPPL.AbstractContext
    inv_temp::Dict{Symbol,T}
    context::Ctx
end

function StreamTemperingContext(
    inv_temp=Dict{:Symbol,Float64}(),
    context=DynamicPPL.DefaultContext(),
)
    return StreamTemperingContext{typeof(inv_temp),typeof(context)}(
        inv_temp, context
    )
end
DynamicPPL.NodeTrait(::StreamTemperingContext) = DynamicPPL.IsParent()
DynamicPPL.childcontext(context::StreamTemperingContext) = context.context
function DynamicPPL.setchildcontext(context::StreamTemperingContext, child)
    return StreamTemperingContext(context.inv_temp, child)
end

get_temp(context::StreamTemperingContext, vn) = get(context.inv_temp, getsym(vn), 1.0)

function DynamicPPL.tilde_assume(context::StreamTemperingContext, right, vn, vi)
    #@info "StreamTemperingContext tilde_assume called for $vn"
    value, logp, vi = DynamicPPL.tilde_assume(context.context, right, vn, vi)
    return value, logp*get_temp(context,vn), vi
end
function DynamicPPL.dot_tilde_assume(context::StreamTemperingContext, right, left, vn, vi)
    #@info "StreamTemperingContext dot_tilde_assume called for $vn"
    value, logp, vi = DynamicPPL.dot_tilde_assume(context.context, right, left, vn, vi)
    return value, logp * get_temp(context,vn), vi
end


function DynamicPPL.tilde_observe(context::StreamTemperingContext, right, left, 
    vn::DynamicPPL.VarName, vi)
    #@info "StreamTemperingContext tilde_observe called vn=$vn"
    # if Main.debug_cnt >= 80
    #     #Main.@infiltrate_main
    #     error("stacktrace")
    # else 
    #     Main.debug_cnt = Main.debug_cnt +1
    # end
    logp, vi = DynamicPPL.tilde_observe(context.context, right, left, vi)
    return logp * get_temp(context,vn), vi
end

function DynamicPPL.dot_tilde_observe(context::StreamTemperingContext, right, left, 
    vn::DynamicPPL.VarName, vi)
    # @info "StreamTemperingContext dot-tilde_observe called vn=$vn"
    # if Main.debug_cnt >= 80
    #     Main.@infiltrate_main
    #     error("stacktrace")
    # else 
    #     Main.debug_cnt = Main.debug_cnt +1
    # end
    logp, vi = DynamicPPL.dot_tilde_observe(context.context, right, left, vn, vi)
    return logp * get_temp(context,vn), vi
end



