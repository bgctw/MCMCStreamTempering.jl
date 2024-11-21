struct ISSampler{space} <: InferenceAlgorithm end
DynamicPPL.getspace(::ISSampler{space}) where {space} = space

ISSampler() = ISSampler{()}()

DynamicPPL.initialsampler(sampler::Sampler{<:ISSampler}) = sampler

function DynamicPPL.initialstep(
    rng::AbstractRNG, model::Model, spl::Sampler{<:ISSampler}, vi::AbstractVarInfo; kwargs...
)
    return Transition(model, vi), nothing
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG, model::Model, spl::Sampler{<:ISSampler}, ::Nothing; kwargs...
)
    vi = VarInfo(rng, model, spl)
    return Transition(model, vi), nothing
end

# Calculate evidence.
function getlogevidence(samples::Vector{<:Transition}, ::Sampler{<:ISSampler}, state)
    return logsumexp(map(x -> x.lp, samples)) - log(length(samples))
end

function DynamicPPL.assume(rng, spl::Sampler{<:ISSampler}, dist::Distribution, vn::VarName, vi)
    if haskey(vi, vn)
        r = vi[vn]
    else
        r = rand(rng, dist)
        vi = push!!(vi, vn, r, dist, spl)
    end
    return r, 0, vi
end

function DynamicPPL.observe(spl::Sampler{<:ISSampler}, dist::Distribution, value, vi)
    return logpdf(dist, value), vi
end