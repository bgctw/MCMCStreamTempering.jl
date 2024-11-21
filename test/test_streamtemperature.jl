using MCMCStreamTempering
using Test
using DynamicPPL
#using AbstractMCMC
using Distributions
using Turing
using DataFrames
#using LinearAlgebra # Diagonal
using ComponentArrays: ComponentArrays as CA
using MCMCStreamTempering: TestUtils as TU

@model function groupmod(o1, o2, m = missing, σ2_o1 = missing, σ2_o2 = missing, ::Type{T} = Float64; 
    length_o1 = 10, size_o2 = (10,8), bias_o2=0.0) where {T}
    if o2 === missing
        o2 = Matrix{T}(undef, size_o2)
    end   
    if o1 === missing
        o1 = Vector{T}(undef, length_o1)
    end        
    n_obs, n_group = size(o2)
    m ~ Normal(0, 3.0)
    σ2_o1 ~ InverseGamma(2, 0.5^2) #same std as bias
    σ2_o2 ~ InverseGamma(2, 0.5^2) #same std as bias
    σ2_r ~ InverseGamma(2, 0.8^2)
    σ_r = √σ2_r
    r = Vector{T}(undef, n_group)
    r .~ Normal(0.0, σ_r)
    # penalize mean of r deviating from zero
    penalty_mean_r = exp(5*abs(mean(r))/σ_r)
    Turing.@addlogprob! penalty_mean_r
    o1 .~ Normal(m, √σ2_o1) # single draw poorly constrained m1
    for i in 1:n_group
        o2[:,i] .~ Normal(m + bias_o2 + r[i], √σ2_o2)
    end
    (;prior=(;m,σ2_o1,σ2_o2,σ2_r,r),obs=(;o1,o2),aux=(; penalty_mean_r))
end
 
v_true = groupmod(missing,missing,0.0, 0.5^2, 0.5^2; bias_o2=0.5)()
ca_true = CA.ComponentVector(v_true)
ca_ind = map(x -> 0, ca_true); ca_ind .= 1:length(ca_ind)
model = groupmod(v_true.obs.o1, v_true.obs.o2)
() -> begin
    v_true.prior
    v_true.prior.r
    v_true.prior.m .+ v_true.prior.r
    v_true.aux
    map(mean, (;v_true.obs..., mr=v_true.prior.m .+ v_true.prior.r, m=v_true.prior.m))

    chain_T0 = sample(model, NUTS(500, 0.65), 400, progress=false);
    hcat(DataFrame(orig=CA.ComponentVector(v_true.prior)), DataFrame(summarystats(chain_T0)))
    # bias in random effects: distributed around 0.5, together with overestimated σ2_r
    # need to penalize mean away? beforehand look at individual samples
    describe(summarystats(chain_T0)[4:end,:mean])
    chn_r = chain_T0[:,4:end,:]
    map(mean, eachrow(Array(chn_r)))

end


@testset "logpriors_var chain" begin
    mod_ctx = StreamTemperingContext(Dict(:o2=>1/(1.0*40)))
    m2 = DynamicPPL.contextualize(model, mod_ctx)
    debug_cnt = 0; chain = sample(m2, NUTS(500, 0.65), 1000, progress=false);
    hcat(DataFrame(orig=CA.ComponentVector(v_true.prior)), DataFrame(summarystats(chain)))
    describe(summarystats(chain)[4:end,:mean])
    chn_r = chain[:,4:end,:]
    map(mean, eachrow(Array(chn_r)))
    genq = generated_quantities(model, chain)

    genq2 = CA.ComponentVector.(vec(genq))
    genq3a = hcat(genq2[1], genq2[2]) 
    genq3 = CA.ComponentMatrix(reduce(hcat, genq2), CA.getaxes(genq3a))
    describe(map(mean, eachrow(genq3[vec(ca_ind[:obs][:o2]),:])))
    map(mean, eachrow(Array(chn_r)))
    
    #keys(v_true.prior)
    #did not converge to true solution
end;

model = TU.basic_imbalanced(TU.v_true.obs.sparse, TU.v_true.obs.rich; c=0.1);

chain = chain_T1 = sample(model, NUTS(500, 0.65), 1000, progress=false);

streams = (;sparse=:sparse, rich=:rich)
streams_p = (;sparse=:p_sparse, rich=:p_rich)
streams_o = (;sparse=:o_sparse, rich=:o_rich)


n_obs = map(s -> length(TU.v_true.obs[s]), streams)
r = 0.2
temp = map(n -> 1 + n*r, n_obs)
#mod_ctx = StreamTemperingContext(Dict(:o_rich=>1/(1.0*40)))
beta = Dict(so => 1/t for (so,t) in zip(streams_o, temp))
mod_ctx = StreamTemperingContext(beta)
chain = chain_T40 = sample(DynamicPPL.contextualize(model, mod_ctx), NUTS(500, 0.65), 1000, progress=false);

() -> begin
    #load_makie_env()
    #using CairoMakie, TwMakieHelpers
    plot_chn(chain)
end
tmp = generated_quantities(model, chain)
# stream = first(streams_p)
pred = map(streams) do stream 
    reduce(hcat, map(x -> x.pred[stream], generated_quantities(model, chain))[:,1])
end;
mean_pred = map(stream -> map(mean, eachrow(pred[stream])), streams)
lower_pred = map(stream -> map(x -> quantile(x, 0.025), eachrow(pred[stream])), streams)
upper_pred = map(stream -> map(x -> quantile(x, 0.975), eachrow(pred[stream])), streams)

tmp_fplot = () -> begin
    #load_makie_env()
    #using CairoMakie, AlgebraOfGraphics, DataFrames, TwMakieHelpers
    ds_sparse = DataFrame(;x=TU.x_sparse, o=TU.v_true.obs.o_sparse, 
        pred=mean_pred[:sparse], lower=lower_pred[:sparse], upper=upper_pred[:sparse])
    ds_rich = DataFrame(;x=TU.x_rich, o=TU.v_true.obs.o_rich,
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
end




