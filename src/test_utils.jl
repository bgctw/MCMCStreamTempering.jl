module TestUtils

using DynamicPPL
using Distributions
using Test
using Random: Random
using StableRNGs
using ComponentArrays: ComponentArrays as CA
using Turing


rng = StableRNG(0817)
x_sparse = sort(rand(rng,10) .+ 0.5)
x_rich = sort(rand(rng,1000) .* 0.3 .+ 0.7)
x = (;sparse=x_sparse, rich=x_rich)

@model function basic_imbalanced(o_sparse, o_rich, a = missing, b = missing, ::Type{T} = Float64;
    x_sparse = x_sparse, x_rich = x_rich, c=0.3, 
    σ_sparse = 0.36, σ_rich = 0.18) where {T}
    if o_sparse === missing
        o_sparse = Vector{T}(undef, length(x_sparse))
    end        
    if o_rich === missing
        o_rich = Vector{T}(undef, length(x_rich))
    end   
    a ~ Uniform(-5.0, +5.0) 
    b ~ Uniform(-5.0, +5.0) 
    p_sparse = a .* x_sparse .+ b .* mean(x_rich)
    p_rich = a .* x_sparse[1] .+ b .* (x_rich .- c)
    o_sparse .~ Normal.(p_sparse, σ_sparse)
    o_rich .~ Normal.(p_rich, σ_rich)
    (;prior=(;a,b,),obs=(;sparse=o_sparse,rich=o_rich),pred=(; sparse=p_sparse, rich=p_rich))
end

v_true = basic_imbalanced(missing, missing, 1.0, 2.0)();
ca_true = CA.ComponentVector(v_true)
ca_ind = map(x -> 0, ca_true); ca_ind .= 1:length(ca_ind)

() -> begin
    # get observation uncertainties as fraction of spread
    # σ_sparse = 0.4 * mean(TU.v_true.aux.p_sparse)
    # σ_rich = 0.3 * mean(TU.v_true.aux.p_rich)
    spread = x -> begin
        ex = extrema(x)
        ex[2] - ex[1]
    end
    σ_sparse = round(0.4 * spread(TU.v_true.aux.sparse); sigdigits=2)
    σ_rich = round(0.3 * spread(TU.v_true.aux.rich); sigdigits=2)
    @show σ_sparse;
    @show σ_rich;
end



end
