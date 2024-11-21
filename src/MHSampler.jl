###
### Sampler states
###

struct MHSampler{space,P} <: InferenceAlgorithm
    proposals::P
end
DynamicPPL.getspace(::MHSampler{space}) where {space} = space


proposal(p::AdvancedMH.Proposal) = p
proposal(f::Function) = AdvancedMH.StaticProposal(f)
proposal(d::Distribution) = AdvancedMH.StaticProposal(d)
proposal(cov::AbstractMatrix) = AdvancedMH.RandomWalkProposal(MvNormal(cov))
proposal(x) = error("proposals of type ", typeof(x), " are not supported")

function MHSampler(space...)
    syms = Symbol[]

    prop_syms = Symbol[]
    props = AMH.Proposal[]

    for s in space
        if s isa Symbol
            # If it's just a symbol, proceed as normal.
            push!(syms, s)
        elseif s isa Pair || s isa Tuple
            # Check to see whether it's a pair that specifies a kernel
            # or a specific proposal distribution.
            push!(prop_syms, s[1])
            push!(props, proposal(s[2]))
        elseif length(space) == 1
            # If we hit this block, check to see if it's
            # a run-of-the-mill proposal or covariance
            # matrix.
            prop = proposal(s)

            # Return early, we got a covariance matrix.
            return MHSampler{(),typeof(prop)}(prop)
        else
            # Try to convert it to a proposal anyways,
            # throw an error if not acceptable.
            prop = proposal(s)
            push!(props, prop)
        end
    end

    proposals = NamedTuple{tuple(prop_syms...)}(tuple(props...))
    syms = vcat(syms, prop_syms)

    return MHSampler{tuple(syms...),typeof(proposals)}(proposals)
end

# Some of the proposals require working in unconstrained space.
transform_maybe(proposal::AMH.Proposal) = proposal
function transform_maybe(proposal::AMH.RandomWalkProposal)
    return AMH.RandomWalkProposal(Bijectors.transformed(proposal.proposal))
end

function MHSampler(model::Model; proposal_type=AMH.StaticProposal)
    priors = DynamicPPL.extract_priors(model)
    props = Tuple([proposal_type(prop) for prop in values(priors)])
    vars = Tuple(map(Symbol, collect(keys(priors))))
    priors = map(transform_maybe, NamedTuple{vars}(props))
    return AMH.MetropolisHastings(priors)
end

#####################
# Utility functions #
#####################

"""
    set_namedtuple!(vi::VarInfo, nt::NamedTuple)

Places the values of a `NamedTuple` into the relevant places of a `VarInfo`.
"""
function set_namedtuple!(vi::DynamicPPL.VarInfoOrThreadSafeVarInfo, nt::NamedTuple)
    # TODO: Replace this with something like
    # for vn in keys(vi)
    #     vi = DynamicPPL.setindex!!(vi, get(nt, vn))
    # end
    for (n, vals) in pairs(nt)
        vns = vi.metadata[n].vns
        nvns = length(vns)

        # if there is a single variable only
        if nvns == 1
            # assign the unpacked values
            if length(vals) == 1
                vi[vns[1]] = [vals[1];]
                # otherwise just assign the values
            else
                vi[vns[1]] = [vals;]
            end
            # if there are multiple variables
        elseif vals isa AbstractArray
            nvals = length(vals)
            # if values are provided as an array with a single element
            if nvals == 1
                # iterate over variables and unpacked values
                for (vn, val) in zip(vns, vals[1])
                    vi[vn] = [val;]
                end
                # otherwise number of variables and number of values have to be equal
            elseif nvals == nvns
                # iterate over variables and values
                for (vn, val) in zip(vns, vals)
                    vi[vn] = [val;]
                end
            else
                error("Cannot assign `NamedTuple` to `VarInfo`")
            end
        else
            error("Cannot assign `NamedTuple` to `VarInfo`")
        end
    end
end

"""
    MHLogDensityFunction

A log density function for the MHSampler sampler.

This variant uses the  `set_namedtuple!` function to update the `VarInfo`.
"""
const MHLogDensityFunction{M<:Model,S<:Sampler{<:MHSampler},V<:AbstractVarInfo} = Turing.LogDensityFunction{
    V,M,<:DynamicPPL.SamplingContext{<:S}
}

function LogDensityProblems.logdensity(f::MHLogDensityFunction, x::NamedTuple)
    # TODO: Make this work with immutable `f.varinfo` too.
    sampler = DynamicPPL.getsampler(f)
    vi = f.varinfo

    x_old, lj_old = vi[sampler], getlogp(vi)
    set_namedtuple!(vi, x)
    vi_new = last(DynamicPPL.evaluate!!(f.model, vi, DynamicPPL.getcontext(f)))
    lj = getlogp(vi_new)

    # Reset old `vi`.
    setindex!!(vi, x_old, sampler)
    setlogp!!(vi, lj_old)

    return lj
end

# unpack a vector if possible
unvectorize(dists::AbstractVector) = length(dists) == 1 ? first(dists) : dists

# possibly unpack and reshape samples according to the prior distribution
reconstruct(dist::Distribution, val::AbstractVector) = DynamicPPL.reconstruct(dist, val)
function reconstruct(dist::AbstractVector{<:UnivariateDistribution}, val::AbstractVector)
    return val
end
function reconstruct(dist::AbstractVector{<:MultivariateDistribution}, val::AbstractVector)
    offset = 0
    return map(dist) do d
        n = length(d)
        newoffset = offset + n
        v = val[(offset+1):newoffset]
        offset = newoffset
        return v
    end
end

"""
    dist_val_tuple(spl::Sampler{<:MHSampler}, vi::VarInfo)

Return two `NamedTuples`.

The first `NamedTuple` has symbols as keys and distributions as values.
The second `NamedTuple` has model symbols as keys and their stored values as values.
"""
function dist_val_tuple(spl::Sampler{<:MHSampler}, vi::DynamicPPL.VarInfoOrThreadSafeVarInfo)
    vns = _getvns(vi, spl)
    dt = _dist_tuple(spl.alg.proposals, vi, vns)
    vt = _val_tuple(vi, vns)
    return dt, vt
end

@generated function _val_tuple(vi::VarInfo, vns::NamedTuple{names}) where {names}
    isempty(names) === 0 && return :(NamedTuple())
    expr = Expr(:tuple)
    expr.args = Any[
        :(
            $name = reconstruct(
                unvectorize(DynamicPPL.getdist.(Ref(vi), vns.$name)),
                DynamicPPL.getval(vi, vns.$name),
            )
        ) for name in names
    ]
    return expr
end

@generated function _dist_tuple(
    props::NamedTuple{propnames}, vi::VarInfo, vns::NamedTuple{names}
) where {names,propnames}
    isempty(names) === 0 && return :(NamedTuple())
    expr = Expr(:tuple)
    expr.args = Any[
        if name in propnames
            # We've been given a custom proposal, use that instead.
            :($name = props.$name)
        else
            # Otherwise, use the default proposal.
            :(
                $name = AMH.StaticProposal(
                    unvectorize(DynamicPPL.getdist.(Ref(vi), vns.$name))
                )
            )
        end for name in names
    ]
    return expr
end

# Utility functions to link
should_link(varinfo, sampler, proposal) = false
function should_link(varinfo, sampler, proposal::NamedTuple{(),Tuple{}})
    # If it's an empty `NamedTuple`, we're using the priors as proposals
    # in which case we shouldn't link.
    return false
end
function should_link(varinfo, sampler, proposal::AdvancedMH.RandomWalkProposal)
    return true
end
# FIXME: This won't be hit unless `vals` are all the exactly same concrete type of `AdvancedMH.RandomWalkProposal`!
function should_link(
    varinfo, sampler, proposal::NamedTuple{names,vals}
) where {names,vals<:NTuple{<:Any,<:AdvancedMH.RandomWalkProposal}}
    return true
end

function maybe_link!!(varinfo, sampler, proposal, model)
    return if should_link(varinfo, sampler, proposal)
        link!!(varinfo, sampler, model)
    else
        varinfo
    end
end

# Make a proposal if we don't have a covariance proposal matrix (the default).
function propose!!(
    rng::AbstractRNG, vi::AbstractVarInfo, model::Model, spl::Sampler{<:MHSampler}, proposal
)
    # Retrieve distribution and value NamedTuples.
    dt, vt = dist_val_tuple(spl, vi)

    # Create a sampler and the previous transition.
    mh_sampler = AMH.MetropolisHastings(dt)
    prev_trans = AMH.Transition(vt, getlogp(vi), false)

    # Make a new transition.
    densitymodel = AMH.DensityModel(
        Base.Fix1(
            LogDensityProblems.logdensity,
            Turing.LogDensityFunction(
                vi,
                model,
                DynamicPPL.SamplingContext(rng, spl, DynamicPPL.leafcontext(model.context)),
            ),
        ),
    )
    trans, _ = AbstractMCMC.step(rng, densitymodel, mh_sampler, prev_trans)

    # TODO: Make this compatible with immutable `VarInfo`.
    # Update the values in the VarInfo.
    set_namedtuple!(vi, trans.params)
    return setlogp!!(vi, trans.lp)
end

# Make a proposal if we DO have a covariance proposal matrix.
function propose!!(
    rng::AbstractRNG,
    vi::AbstractVarInfo,
    model::Model,
    spl::Sampler{<:MHSampler},
    proposal::AdvancedMH.RandomWalkProposal,
)
    # If this is the case, we can just draw directly from the proposal
    # matrix.
    vals = vi[spl]

    # Create a sampler and the previous transition.
    mh_sampler = AMH.MetropolisHastings(spl.alg.proposals)
    prev_trans = AMH.Transition(vals, getlogp(vi), false)

    # Make a new transition.
    #Main.@infiltrate_main
    densitymodel = AMH.DensityModel(
        Base.Fix1(
            LogDensityProblems.logdensity,
            Turing.LogDensityFunction(
                vi,
                model,
                DynamicPPL.SamplingContext(rng, spl, DynamicPPL.leafcontext(model.context)),
            ),
        ),
    )
    trans, _ = AbstractMCMC.step(rng, densitymodel, mh_sampler, prev_trans)

    return setlogp!!(DynamicPPL.unflatten(vi, spl, trans.params), trans.lp)
end

function DynamicPPL.initialstep(
    rng::AbstractRNG,
    model::AbstractModel,
    spl::Sampler{<:MHSampler},
    vi::AbstractVarInfo;
    kwargs...,
)
    # If we're doing random walk with a covariance matrix,
    # just link everything before sampling.
    vi = maybe_link!!(vi, spl, spl.alg.proposals, model)

    return Transition(model, vi), vi
end

function AbstractMCMC.step(
    rng::AbstractRNG, model::Model, spl::Sampler{<:MHSampler}, vi::AbstractVarInfo; kwargs...
)
    # Cases:
    # 1. A covariance proposal matrix
    # 2. A bunch of NamedTuples that specify the proposal space
    vi = propose!!(rng, vi, model, spl, spl.alg.proposals)

    return Transition(model, vi), vi
end

####
#### Compiler interface, i.e. tilde operators.
####
function DynamicPPL.assume(rng, spl::Sampler{<:MHSampler}, dist::Distribution, vn::VarName, vi)
    error("assume method called instead of corresponding tilde_assume for vn=$vn")
    #Main.@infiltrate_main
    DynamicPPL.updategid!(vi, vn, spl)
    r = vi[vn]
    return r, logpdf_with_trans(dist, r, istrans(vi, vn)), vi
end

function DynamicPPL.tilde_assume(
    ::DynamicPPL.IsLeaf, rng::Random.AbstractRNG, context::DynamicPPL.AbstractContext, spl::Sampler{<:MHSampler}, dist::Distribution, vn::VarName, vi
)
    #Main.@infiltrate_main
    r0 = copy(vi[vn])
    DynamicPPL.updategid!(vi, vn, spl)
    r = vi[vn]
    @show r0, r
    #lpo = logpdf_with_trans(dist, r, istrans(vi, vn))
    r1, lp, vi1 = tilde_assume(context, dist, vn, vi)
    return r, lp, vi
end

function DynamicPPL.dot_assume(
    rng,
    spl::Sampler{<:MHSampler},
    dist::MultivariateDistribution,
    vn::VarName,
    var::AbstractMatrix,
    vi,
)
    error("dot_assume Matrix method called instead of corresponding tilde_assume for vn=$vn")
    @assert dim(dist) == size(var, 1)
    getvn = i -> VarName(vn, vn.indexing * "[:,$i]")
    vns = getvn.(1:size(var, 2))
    DynamicPPL.updategid!.(Ref(vi), vns, Ref(spl))
    r = vi[vns]
    var .= r
    return var, sum(logpdf_with_trans(dist, r, istrans(vi, vns[1]))), vi
end
function DynamicPPL.dot_assume(
    rng,
    spl::Sampler{<:MHSampler},
    dists::Union{Distribution,AbstractArray{<:Distribution}},
    vn::VarName,
    var::AbstractArray,
    vi,
)
    error("dot_assume Array method called instead of corresponding tilde_assume for vn=$vn")
    getvn = ind -> VarName(vn, vn.indexing * "[" * join(Tuple(ind), ",") * "]")
    vns = getvn.(CartesianIndices(var))
    DynamicPPL.updategid!.(Ref(vi), vns, Ref(spl))
    r = reshape(vi[vec(vns)], size(var))
    var .= r
    return var, sum(logpdf_with_trans.(dists, r, istrans(vi, vns[1]))), vi
end

function DynamicPPL.observe(spl::Sampler{<:MHSampler}, d::Distribution, value, vi)
    #Main.@infiltrate_main
    #TODO replace by tilde_observe childcontext
    tmp = DynamicPPL.observe(SampleFromPrior(), d, value, vi)
    tmp2 = DynamicPPL.tilde_observe(DefaultContext(), d, value, vi)
    return DynamicPPL.observe(SampleFromPrior(), d, value, vi)
end

function DynamicPPL.dot_observe(
    spl::Sampler{<:MHSampler},
    ds::Union{Distribution,AbstractArray{<:Distribution}},
    value::AbstractArray,
    vi,
)
    return DynamicPPL.dot_observe(SampleFromPrior(), ds, value, vi)
end