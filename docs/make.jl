using MCMCStreamTempering
using Documenter

DocMeta.setdocmeta!(MCMCStreamTempering, :DocTestSetup, :(using MCMCStreamTempering); recursive=true)

makedocs(;
    modules=[MCMCStreamTempering],
    authors="Thomas Wutzler <twutz@bgc-jena.mpg.de> and contributors",
    sitename="MCMCStreamTempering.jl",
    format=Documenter.HTML(;
        canonical="https://bgctw.github.io/MCMCStreamTempering.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/bgctw/MCMCStreamTempering.jl",
    devbranch="main",
)
