"""
Context that multiplies each log-prior by mod
used to test whether varwise_logpriors respects child-context.
"""
struct TestLogModifyingChildContext{T,Ctx} <: DynamicPPL.AbstractContext
    mod::T
    context::Ctx
end
function TestLogModifyingChildContext(
    mod=1.2,
    context::DynamicPPL.AbstractContext=DynamicPPL.DefaultContext(),
        #OrderedDict{VarName,Vector{Float64}}(),PriorContext()),
)
    return TestLogModifyingChildContext{typeof(mod),typeof(context)}(
        mod, context
    )
end
DynamicPPL.NodeTrait(::TestLogModifyingChildContext) = DynamicPPL.IsParent()
DynamicPPL.childcontext(context::TestLogModifyingChildContext) = context.context
function DynamicPPL.setchildcontext(context::TestLogModifyingChildContext, child)
    return TestLogModifyingChildContext(context.mod, child)
end

function DynamicPPL.tilde_assume(context::TestLogModifyingChildContext, right, vn, vi)
    #@info "TestLogModifyingChildContext tilde_assume called for $vn"
    value, logp, vi = DynamicPPL.tilde_assume(context.context, right, vn, vi)
    return value, logp*context.mod, vi
end
function DynamicPPL.dot_tilde_assume(context::TestLogModifyingChildContext, right, left, vn, vi)
    #@info "TestLogModifyingChildContext dot_tilde_assume called for $vn"
    value, logp, vi = DynamicPPL.dot_tilde_assume(context.context, right, left, vn, vi)
    return value, logp*context.mod, vi
end

function DynamicPPL.tilde_observe(context::TestLogModifyingChildContext, right, left, 
    vn::DynamicPPL.VarName, vi)
    #@info "TestLogModifyingChildContext tilde_observe called vn=$vn"
    # if Main.debug_cnt >= 80
    #     #Main.@infiltrate_main
    #     error("stacktrace")
    # else 
    #     Main.debug_cnt = Main.debug_cnt +1
    # end
    logp, vi = DynamicPPL.tilde_observe(context.context, right, left, vi)
    return logp .* context.mod, vi
end
function DynamicPPL.tilde_observe(context::TestLogModifyingChildContext, right, left, vi)
    error("TestLogModifyingChildContext tilde_observe called without vn")
end
function DynamicPPL.tilde_observe(
    context::TestLogModifyingChildContext, sampler::AbstractMCMC.AbstractSampler, right, left, 
    vi)
    error("TestLogModifyingChildContext tilde_observe-sampler called without vn")
end
function DynamicPPL.tilde_observe(
    context::TestLogModifyingChildContext, sampler::AbstractMCMC.AbstractSampler, right, left, 
    vn::DynamicPPL.VarName, vi)
    #@info "TestLogModifyingChildContext tilde_observe-sampler called vn=$vn"
    # if Main.debug_cnt >= 80
    #     #Main.@infiltrate_main
    #     error("stacktrace")
    # else 
    #     Main.debug_cnt = Main.debug_cnt +1
    # end
    logp, vi = DynamicPPL.tilde_observe(context.context, sampler, right, left, vi)
    return logp .* context.mod, vi
end

function DynamicPPL.dot_tilde_observe(context::TestLogModifyingChildContext, right, left, vi)
    error("TestLogModifyingChildContext dot_tilde_observe called without vn")
end
function DynamicPPL.dot_tilde_observe(context::TestLogModifyingChildContext, right, left, 
    vn::DynamicPPL.VarName, vi)
    #@info "TestLogModifyingChildContext dot-tilde_observe called vn=$vn"
    # if Main.debug_cnt >= 80
    #     #Main.@infiltrate_main
    #     error("stacktrace")
    # else 
    #     Main.debug_cnt = Main.debug_cnt +1
    # end
    logp, vi = DynamicPPL.dot_tilde_observe(context.context, right, left, vn, vi)
    return logp .* context.mod, vi
end
function DynamicPPL.dot_tilde_observe(
    context::TestLogModifyingChildContext, sampler::AbstractMCMC.AbstractSampler, 
    right, left, vi)
    error("TestLogModifyingChildContext dot_tilde_observe-sampler called without vn")
end



# simple Sampler to explore


