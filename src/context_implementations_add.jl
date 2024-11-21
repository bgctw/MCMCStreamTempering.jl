function DynamicPPL.tilde_observe!!(context::SamplingContext, right, left, 
    vname::DynamicPPL.VarName, vi)
    # pass down vname
    return DynamicPPL.tilde_observe!!(context.context, right, left, vname, vi)
end

# replacing current method in DynamicPPL to pass down vname
#__precompile__(false)
function DynamicPPL.tilde_observe!!(context, right, left, vname::DynamicPPL.VarName, vi)
    # pass down vname
    logp, vi = DynamicPPL.tilde_observe(context, right, left, vname, vi)
    return left, DynamicPPL.acclogp_observe!!(context, vi, logp)
end

# if passing down vname, here drop it, because not any more under context control
function DynamicPPL.tilde_observe(
    ::DynamicPPL.IsLeaf, context::DynamicPPL.AbstractContext, right, left, 
    vname::DynamicPPL.VarName, vi) 
    DynamicPPL.observe(right,left,vi)
end


function DynamicPPL.dot_tilde_observe!!(context::SamplingContext, right, left, 
    vname::DynamicPPL.VarName, vi)
    # pass down vname
    return DynamicPPL.dot_tilde_observe!!(context.context, right, left, vname, vi)
end

# replacing current method in DynamicPPL to pass down vname
function DynamicPPL.dot_tilde_observe!!(context, right, left, vname::DynamicPPL.VarName, vi)
    logp, vi = DynamicPPL.dot_tilde_observe(context, right, left, vname, vi)
    return left, DynamicPPL.acclogp_observe!!(context, vi, logp)
end

# if passing down vname, here drop it, because not any more under context control
function DynamicPPL.dot_tilde_observe(
    ::DynamicPPL.IsLeaf, context::DynamicPPL.AbstractContext, right, left, 
    vname::DynamicPPL.VarName, vi) 
    DynamicPPL.dot_observe(right,left,vi)
end
