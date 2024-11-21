module MCMCStreamTempering
using DynamicPPL
using DynamicPPL: Metadata, VarInfo, TypedVarInfo,
    islinked, invlink!, link!,
    setindex!!, push!!,
    setlogp!!, getlogp,
    VarName, getsym, 
    _getvns, getdist,
    Model, Sampler, SampleFromPrior, SampleFromUniform,
    DefaultContext, PriorContext,
    LikelihoodContext, set_flag!, unset_flag!,
    getspace, inspace
using AbstractMCMC
using Distributions
using Random
using Turing
using Turing.Inference: InferenceAlgorithm, Transition, AbstractModel
using AdvancedMH: AdvancedMH
using AdvancedMH: AdvancedMH as AMH
using LogDensityProblems: LogDensityProblems
using Loess
using ComponentArrays: ComponentArrays as CA

export StreamTemperingContext
include("streamtemperingcontext.jl")

export TestLogModifyingChildContext
include("testcontext.jl")

export estimate_obs_unc, estimate_prop_weights_model_unc, compute_T_streams, estimate_model_discrepancy
include("temper_model_error.jl")

include("context_implementations_add.jl")

include("ISSampler.jl")
include("MHSampler.jl")

include("test_utils.jl")
end
