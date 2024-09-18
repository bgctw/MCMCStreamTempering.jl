using MCMCStreamTempering
using Test
using Aqua

@testset "MCMCStreamTempering.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(MCMCStreamTempering)
    end
    # Write your tests here.
end
