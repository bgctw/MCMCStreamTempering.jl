using Loess: Loess


"""
    NamedNTuple{T}

Shorthand for NamedTuple with Tuple type being an NTuple with eltype T`.
"""
const NamedNTuple{T,N,S} = NamedTuple{S,NTuple{N,T}} where {T, S, N}

"""
    NamedNTupleOrComponentVector{T}

Shorthand for Union of `NamedNTuple{T}` or `ComponentVector{T}` of `eltype(T)`.
"""
const NamedNTupleOrComponentVector{T} = Union{NamedNTuple{T}, CA.ComponentVector{T}} where {T}

"""
    estimate_obs_unc(x,y; span=0.8)

Estimate std-deviation of observation error in a series by inspecting
the residuals of a loess model `y ~ x`.
Keyword argument `span` controls the smoothmess of the fit. A value of 1 corresponds
to a straight line, 0.8 is adequate for a single maximum.
The fit can be inspected by plotting `pred ~ x` in the same plot as `y ~ x`.

## Return value is a `NamedTuple`
- `std_obsunc`: estimated standard deviation of observation uncertainty
- `SS_obsunc`: sum of squared residuals of observation uncertainty
- `pred`: predictions by the loess model, useful to inspect smoothness.
"""
function estimate_obs_unc(
    x::AbstractVector{<:Number}, y::AbstractVector{<:Number}; span=0.8)
    model = Loess.loess(x, y; span)
    pred = Loess.predict(model, x)
    resid = pred .- y
    SS_obsunc = sum(abs2, resid)
    #fig, _= plot(x, y);
    #lines!(x, pred); display(fig)
    #tmp = plot!(pl, x, resid)
    (;std_obsunc = std(resid), SS_obsunc, pred)
    # n_eff = compute_effective_n_obs(resid, x)
    # n_eff < 10 && Main.@infiltrate_main #error("Expected n_eff >= 10 but was $n_eff. Mixed repicates or forgot to demean?")

    # std_eff_val = compute_std_eff(resid, x; n_eff)
    # SSo = sum(abs2, resid)
    # (; n_fin=count_finite_obs(y), n_eff, std_eff=std_eff_val,
    #     SSo, mean_var_obs=var_cor(resid; n_eff)
    # )
end
function estimate_obs_unc(
    xs::NamedNTuple{<:AbstractVector{<:Number}}, ys::NamedNTuple{<:AbstractVector{<:Number}}; spans::NamedNTuple{<:Number} = map(xi -> 0.8, x))
    streams = keys(xs)
    tmp = (; map(streams) do s 
        s, estimate_obs_unc(xs[s], ys[s], span=spans[s]);    
    end...)
    vars = (;zip(keys(tmp[1]), keys(tmp[1]))...)
    ret = map(vari -> map(x -> x[vari], tmp), vars)    
    return(ret)
end


"""
    estimate_prop_weights_model_unc(obs_streams, std_eff_streams)

Estimate ratio of `spread / stddev(obs_error)`, where spread is the 95% interval of
observed value. 

This is helpful for allowing larger relative model discrepancy (per observation uncertainty) for the streams with a larger spread per observation uncertainty.

## Arguments
- `obs_streams`: `ComponentVector` or `NamedTuple` `stream -> observations`
- `std_eff_streams`: `ComponentVector` or `NamedTuple` `stream -> std_dev(obs_error)`

## Result NamedTuple with ComponentVectors 
- `w`: multiplicators of allowed model discrepancy. Specifically `w0 ./ minimum(w0)`
- `w0`: original ratios `spread / stddev(obs_error)`
"""
function estimate_prop_weights_model_unc(obs_streams, std_eff_streams)
    #std_effs = std_eff_streams[keys(obs_streams)]
    #s = first(keys(obs_streams))
    w0 = CA.ComponentVector(;
        map(keys(obs_streams)) do s
            obs_s = obs_streams[s]
            std_resid = std_eff_streams[s]
            (Symbol(s), estimate_prop_weights_model_unc(obs_s, std_resid))
        end...)
    w = w0 ./ minimum(w0)
    (;w, w0)
end
function estimate_prop_weights_model_unc(x::AbstractVector, std_resid::Number)
    ex = quantile.(Ref(x), (0.025, 0.975))
    σ2am = abs2((ex[2] - ex[1])) 
    σ2am / abs2(std_resid)
end

# """
#     temper_unc(Type{<:Distribution}, mean, unc, T)

# Return updated uncertainty uncertainty parameter so that  Log-Likekilood L 
# (excluding the normalizing factor): 
# `L(x; unc_new) = L(x; unc)/T`.
# For the Normal and LogNormal distribution this is `σ * sqrt(T)`
# For the MvNormal and MvLogNormal this is sqrt(T)I * Σ * sqrt(T)I)
# """
# temper_unc(::Type{Normal}, mean, σ, T) = σ * sqrt(T)
# function temper_unc(::Type{MvNormal}, mean, Σ, T)
#     sTI = UniformScaling(sqrt(T))
#     sTI * Σ * sTI
# end
# temper_unc(::Type{LogNormal}, mean, σ, T) = σ * sqrt(T)
# function temper_unc(::Type{MvLogNormal}, mean, Σ, T)
#     sTI = UniformScaling(sqrt(T))
#     sTI * Σ * sTI
# end

"""
Compute temperature of data streams for given ratio of `r = model_discrepancy / stddev(obs_error)`.


"""
function compute_T_streams(obs::NamedNTuple{AbstractArray<:Number}, r; 
    kwargs...)
    n_obs = map(count_finite_obs, obs) # map does not work with ComponentVector
    compute_T_streams(n_obs, r; kwars...)
end
function compute_T_streams(cnt_obs::NamedNTupleOrComponentVector{<:Integer}, r;w_streams::NamedNTupleOrComponentVector{<:Number})
    streams = keys(cnt_obs)
    CA.ComponentVector(; map(streams) do s
        T_s = compute_T_streams(cnt_obs[s], r, w_streams[s])
        (s, T_s)
    end...)
end
compute_T_streams(cnt_obs, r, w) = 1 + cnt_obs * r * w

function estimate_model_discrepancy(SS_resids::AbstractVector, SS_obserrs::AbstractVector; kwargs...) 
    estimate_model_discrepancy.(SS_resid, SS_obserr; kwargs...)
end

estimate_model_discrepancy(SS_resid::Number, SS_obserr::Number; w_stream::Number) = (SS_resid / SS_obserr - 1) / w_stream


count_finite_obs(obs::AbstractVector) = sum(skipmissing(isfinite.(obs)))



