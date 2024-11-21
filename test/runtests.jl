using Test, SafeTestsets
const GROUP = get(ENV, "GROUP", "All") # defined in in CI.yml
@show GROUP

@time begin
    if GROUP == "All" || GROUP == "Basic"
        #@safetestset "Tests" include("test/test_modifyingchildcontext.jl")
        @time @safetestset "test_modifyingchildcontext" include("test_modifyingchildcontext.jl")
        #@safetestset "Tests" include("test/test_streamtemperature.jl")
        @time @safetestset "test_streamtemperature" include("test_streamtemperature.jl")
    end
    # TODO look into JET failing and activate again
    # if GROUP == "All" || GROUP == "JET"
    #     #@safetestset "Tests" include("test/test_JET.jl")
    #     @time @safetestset "test_JET" include("test_JET.jl")
    #     #@safetestset "Tests" include("test/test_aqua.jl")
    #     @time @safetestset "test_Aqua" include("test_aqua.jl")
    # end
end

