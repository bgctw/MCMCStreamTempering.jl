using MCMCStreamTempering
using MCMCStreamTempering: MCMCStreamTempering as CP
using Test
using DynamicPPL
#using AbstractMCMC
using Distributions
using Turing
#using LinearAlgebra # Diagonal
using ComponentArrays: ComponentArrays as CA
using MCMCStreamTempering: TestUtils as TU

streams = (;sparse=:sparse, rich=:rich)
streams_o = (;sparse=:o_sparse, rich=:o_rich)
spans = (;sparse=1.0, rich=1.0)

obs = TU.v_true.obs
std_obsunc, SS_obsunc, pred = estimate_obs_unc(TU.x, obs; spans);

() -> begin
    #using CairoMakie
    s = :sparse
    fig, _ = plot(TU.x[s], obs[s]);
    lines!(TU.x[s], pred[s])
    display(fig)
end

w_streams, w0 = estimate_prop_weights_model_unc(obs, std_obsunc)

n_obs = map(CP.count_finite_obs, obs)
r = 0.3 # initial estimate will be adjusted on first sample below
#r = 0.2 from below
temp = compute_T_streams(n_obs, r; w_streams)
#mod_ctx = StreamTemperingContext(Dict(:o_rich=>1/(1.0*40)))
beta = Dict(so => 1/t for (so,t) in zip(streams_o, temp))
mod_ctx = StreamTemperingContext(beta)
model = TU.basic_imbalanced(obs.sparse, obs.rich; c=0.1);
chain = chain_r02 = sample(DynamicPPL.contextualize(model, mod_ctx), NUTS(500, 0.65), 1000, progress=false);
#chain = chain_r0 = sample(model, NUTS(500, 0.65), 1000, progress=false);

() -> begin
    #load_makie_env()
    #using CairoMakie, TwMakieHelpers
    plot_chn(chain)
end


pred = map(streams) do stream 
    reduce(hcat, map(x -> x.pred[stream], generated_quantities(model, chain))[:,1])
end;
mean_pred = map(stream -> map(mean, eachrow(pred[stream])), streams)
lower_pred = map(stream -> map(x -> quantile(x, 0.025), eachrow(pred[stream])), streams)
upper_pred = map(stream -> map(x -> quantile(x, 0.975), eachrow(pred[stream])), streams)


# compute discrepancy to stddev_obs ratio
rs_streams = map(streams) do s 
    pred_s = pred[s]
    resid_s = obs[s] .- pred_s 
    SS_resid = map(eachcol(resid_s)) do resid_si
        sum(abs2, resid_si)
    end
    estimate_model_discrepancy.(SS_resid, SS_obsunc[s]; w_stream=w_streams[s])
end
rs_max = map(maximum, eachrow(hcat(rs_streams...)));
describe(rs_max)
r = minimum(rs_max)


#load_makie_env()
using CairoMakie, AlgebraOfGraphics, DataFrames, TwMakieHelpers
ds_sparse = DataFrame(;x=TU.x_sparse, o=TU.v_true.obs.sparse, 
    pred=mean_pred[:sparse], lower=lower_pred[:sparse], upper=upper_pred[:sparse])
ds_rich = DataFrame(;x=TU.x_rich, o=TU.v_true.obs.rich,
    pred=mean_pred[:rich], lower=lower_pred[:rich], upper=upper_pred[:rich])
ds = vcat(
    (ds -> begin; ds[!, :scenario] .= "sparse"; ds; end)(ds_sparse),
    (ds -> begin; ds[!, :scenario] .= "rich"; ds; end)(ds_rich),
 )
p1 = data(ds) * mapping(:x, layout=:scenario) * 
    (mapping(:o) * visual(Scatter, color = (:black, 0.6)) + 
    mapping(:lower, :upper) * visual(Band, color = (:blue, 0.6)) +
    mapping(:pred) * visual(Lines, color = (:blue, 0.9)))
draw(p1, facet=(; linkxaxes=:none, linkyaxes=:none))




