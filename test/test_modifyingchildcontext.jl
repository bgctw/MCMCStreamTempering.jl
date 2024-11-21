using MCMCStreamTempering
using Test
using DynamicPPL
#using AbstractMCMC
using Distributions
using Turing
#using LinearAlgebra # Diagonal

@model function demo_obsserveloop(x, ::Type{TV} = Vector{Float64}) where {TV}
    s ~ InverseGamma(2, 3)
    m = TV(undef, length(x))
    for i in eachindex(x)
        m[i] ~ Normal(0, √s)
        x[i] ~ Normal(m[i], √s)
    end
    #x ~ MvNormal(m, √s)
    #x .~ Normal.(m, √s)
end
@model function demo_obsservedot(x, ::Type{TV} = Vector{Float64}) where {TV}
    s ~ InverseGamma(2, 3)
    m = TV(undef, length(x))
    m .~ Normal(0, √s)
    x .~ Normal.(m, √s)
end
x_true = [0.3290767977680923, 0.038972110187911684, -0.5797496780649221]
model = demo_obsserveloop(x_true)
model = demo_obsservedot(x_true)

@testset "logpriors_var chain" begin
    mod_ctx = TestLogModifyingChildContext(1.2)
    m2 = DynamicPPL.contextualize(model, mod_ctx)
    debug_cnt = 0; chain = sample(m2, NUTS(), 100, progress=false);
end;





