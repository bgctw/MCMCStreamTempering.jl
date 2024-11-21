using MCMCStreamTempering
using Test
using Aqua

@testset "Code quality (Aqua.jl)" begin
    Aqua.test_all(MCMCStreamTempering;
        #unbound_args = false, # does not recognize Union{NTuple{N, Symbol}
        stale_deps = (ignore = [:Requires],),
        #ambiguities = false
        )
end;

# @testset "ambiguities package" begin
#     Aqua.test_ambiguities(TwMakieHelpers;)
# end;
