using DataFrames, CSV
using TidierData
using Random, StatsBase, Distributions
using DifferentialEquations 
using Turing, LogDensityProblems
using MCMCChains, KernelDensity
using SavitzkyGolay # filter
using Gnuplot, PrettyTables, TerminalPager, Match
using Dates, Parameters, Configurations, TOML, GarishPrint
Gnuplot.options.term = "sixelgd truecolor enhanced size 1000,400 scroll font arial 10"
# when adding new packages, avoid precompilation by
# pkg> add --preserve=all Example 
# and see https://pkgdocs.julialang.org/dev/environments/#Controlling-Auto-precompilation

@option struct ProvinceParams
    popsize::Int
    immunity_flu::Float64
    immunity_cov::Float64
    immunity_rsv::Float64
    t_school_flu::Float64
    t_school_rsv::Float64
    t_school_cov::Float64
    n_windin_flu::Int
    n_windin_cov::Int
    n_windin_rsv::Int
end
Base.show(io::IO, ::MIME"text/plain", x::ProvinceParams) = GarishPrint.pprint_struct(io, x)

# helper functions to apply the savitzky-golay in the TidierData pipeline.
function make_incidence(X) 
    y = similar(X)
    y .= X - circshift(X, 1)    
    y[1] = y[2]
    return y
end
apply_sg(vec) = savitzky_golay(vec, 7, 1).y
logistic(x) = 1.0 / (1.0 + exp(-x))

function read_province(name) 
    # unable to do: New Brunswick and Sask 
    # Newfoundland, PEI have slightly less data. 
    println("getting data for province: $name")
     # Parse the whole file into a nested dictionary
    data = TOML.parsefile("parameters.toml")
    _tomlname = replace(name, " " => "-")
    !haskey(data, _tomlname) && error("Province '$(_tomlname)' not found")
    params_obj = from_dict(ProvinceParams, data[_tomlname]) # convert Configurations struct

    # data from: https://health-infobase.canada.ca/wastewater/#download
    df = DataFrame(CSV.File("canada_wastewater.csv"))
    # clean up a little 
    df = @chain df begin
        @mutate(weekstart = Date(weekstart, "mm/dd/yyyy"))
        @filter(province == !!name)
        @filter(ismissing(site) && ismissing(city))
        @filter(measureid in ("fluA", "fluB", "covN2", "rsv"))
        # use epiweek 26 if using WIND IN period of 9
        @filter(EpiYear == 2024 && EpiWeek >= 26 || EpiYear == 2025 && EpiWeek <= 34) 
        @select(pruid, site, city, province, weekstart, EpiYear, EpiWeek, measureid, w_avg, populationcoverage)
        @pivot_wider(df, names_from = measureid, values_from = w_avg)
        @mutate(flu = fluA + fluB) # combine fluA and fluB into one signal.
        @mutate(cov_sg = ~apply_sg(covN2)) # the ~ means not to vectorize. 
        @mutate(flu_sg = ~apply_sg(flu)) # the ~ means not to vectorize. 
        @mutate(rsv_sg = ~apply_sg(rsv)) # the ~ means not to vectorize. 
    end

    # plot the data
    @gp "reset" 
    @gp :- "set multiplot layout 1, 3" :- # 1 row, 3 columns
    @gp :- 1 1:nrow(df) df.covN2 "with lines title 'covN2' lc rgb 'black'" :- 
    @gp :- 1 1:nrow(df) df.cov_sg "with lines title 'covN2_sg' lc rgb 'blue'" :-
    @gp :- 2 1:nrow(df) df.flu "with lines title 'flu' lc rgb 'black'" :- 
    @gp :- 2 1:nrow(df) df.flu_sg "with lines title 'flu_sg' lc rgb 'blue'" :-
    @gp :- 3 1:nrow(df) df.rsv "with lines title 'rsv' lc rgb 'black'" :- 
    @gp :- 3 1:nrow(df) df.rsv_sg "with lines title 'rsv_sg' lc rgb 'blue'" :- 
    println("plot set, call using @gp in terminal to display.") 

    println("adjusted pop size: $(params_obj.popsize)")
    # print LOD week starts. We are assuming LOD < 1 is no signal. 
    # WW data is in weeks, so convert to days.
    lod_week_start_flu = find_emergence_week(df.flu_sg, 1.0) 
    lod_week_start_cov = find_emergence_week(df.cov_sg, 1.0)
    lod_week_start_rsv = find_emergence_week(df.rsv_sg, 1.0)
    println("LOD week start (threshold: 1.0): 
    flu: $lod_week_start_flu (in days: $(lod_week_start_flu * 7 - 6)), 
    covid: $lod_week_start_cov (in days: $(lod_week_start_cov * 7 - 6)), 
    rsv: $lod_week_start_rsv (in days: $(lod_week_start_rsv * 7 - 6))
        Use information to set the correct wind-in period in province parameters")

    println("SG data: 
    flu: $(round.(df.flu_sg, digits=2)),

    covid: $(round.(df.cov_sg, digits=2)),

    rsv: $(round.(df.rsv_sg, digits=2)))")

    return df::DataFrame, params_obj
end

function hyperparameters(name)
    _, params = read_province(name) 

    @match name begin
    "Ontario" => (
    βᵢ_base = truncated(Normal(0.35, 0.10), 0.15, 0.42),
    βᵢ_peak = truncated(Normal(1.05, 0.15), 0.50, 1.50),
    βᵥ_base = truncated(Normal(0.50, 0.10), 0.30, 0.75),
    βᵥ_peak = truncated(Normal(0.87, 0.12), 0.60, 1.30),
    βᵣ_base = truncated(Normal(0.50, 0.10), 0.25, 0.72),
    βᵣ_peak = truncated(Normal(1.05, 0.12), 0.55, 1.10),
    i0c = truncated(Normal(200.0, 500.0), 10.0, params.popsize * 0.05),
    ε_flu = truncated(Exponential(1.0), 0.1, 5.0),
    ε_cov = truncated(Exponential(0.05), 0.0, 0.1),
    ε_rsv = truncated(Exponential(3.0), 0.5, 10.0)
    )
    "British Columbia" => (
    βᵢ_base = truncated(Normal(0.35, 0.10), 0.15, 0.42),
    βᵢ_peak = truncated(Normal(1.05, 0.15), 0.50, 1.50),
    #βᵥ_base = truncated(Normal(0.45, 0.08), 0.25, 0.62),
    #βᵥ_peak = truncated(Normal(0.45, 0.08), 0.25, 0.62),
    βᵥ_base = truncated(Normal(0.45, 0.08), 0.25, 0.62),   # sub-threshold summer
    βᵥ_peak = truncated(Normal(0.87, 0.12), 0.60, 1.30),   # above-threshold winter
    βᵣ_base = truncated(Normal(0.50, 0.10), 0.25, 0.72),
    βᵣ_peak = truncated(Normal(1.05, 0.12), 0.55, 1.10),
    i0c = truncated(Normal(4000.0, 800.0), 500.0, params.popsize * 0.10),
    ε_flu = truncated(Exponential(1.0), 0.1, 5.0),
    ε_cov = truncated(Exponential(0.05), 0.0, 0.1),
    ε_rsv = truncated(Exponential(3.0), 0.5, 10.0)
    )
    "Alberta" => (
    βᵢ_base = truncated(Normal(0.35, 0.10), 0.15, 0.42),
    βᵢ_peak = truncated(Normal(1.05, 0.15), 0.50, 1.50),        
    βᵥ_base = truncated(Normal(0.65, 0.05), 0.55, 0.75),   # above threshold — COVID growing
    βᵥ_peak = truncated(Normal(0.87, 0.12), 0.60, 1.30),   # winter amplification
    βᵣ_base = truncated(Normal(0.50, 0.10), 0.25, 0.72),
    βᵣ_peak = truncated(Normal(1.05, 0.12), 0.55, 1.10),
    i0c = truncated(Normal(300.0, 150.0), 50.0, 2000.0),
    ε_flu = truncated(Exponential(1.0), 0.1, 5.0),
    ε_cov = truncated(Exponential(0.05), 0.0, 0.1),
    ε_rsv = truncated(Exponential(3.0), 0.5, 10.0)
    )
    "Manitoba" => (
    βᵢ_base = truncated(Normal(0.35, 0.10), 0.15, 0.42),
    βᵢ_peak = truncated(Normal(1.05, 0.15), 0.65, 1.50),
    βᵥ_base = truncated(Normal(0.68, 0.05), 0.58, 0.78),   # above threshold — COVID growing
    βᵥ_peak = truncated(Normal(0.87, 0.12), 0.60, 1.30),
    βᵣ_base = truncated(Normal(0.50, 0.10), 0.25, 0.72),
    βᵣ_peak = truncated(Normal(1.05, 0.12), 0.55, 1.10),
    i0c     = truncated(Normal(800.0, 200.0), 100.0, params.popsize * 0.10),
    ε_flu = truncated(Exponential(1.0), 0.1, 5.0),
    ε_cov = truncated(Exponential(0.05), 0.0, 0.1),
    ε_rsv = truncated(Exponential(3.0), 0.5, 10.0)
    )
    "Quebec" => (
    βᵢ_base = truncated(Normal(0.35, 0.10), 0.15, 0.42),
    βᵢ_peak = truncated(Normal(1.05, 0.15), 0.65, 1.50),
    βᵥ_base = truncated(Normal(0.664, 0.003), 0.655, 0.672),
    βᵥ_peak = truncated(Normal(1.05, 0.08), 0.90, 1.20),
    βᵣ_base = truncated(Normal(0.50, 0.10), 0.25, 0.72),
    βᵣ_peak = truncated(Normal(1.05, 0.12), 0.55, 1.10),
    i0c = truncated(Normal(3000.0, 500.0), 500.0, params.popsize * 0.05),
    ε_flu = truncated(Exponential(1.0), 0.1, 5.0),
    ε_cov = truncated(Exponential(5.0), 1.0, 40.0), # quebec
    ε_rsv = truncated(Exponential(3.0), 0.5, 10.0)
    )
    "Nova Scotia" => (
    βᵢ_base = truncated(Normal(0.35, 0.10), 0.15, 0.42),
    βᵢ_peak = truncated(Normal(1.05, 0.15), 0.65, 1.50),
    βᵥ_base = truncated(Normal(0.45, 0.08), 0.25, 0.62),   # sub-threshold — COVID declining after week 5
    #βᵥ_peak = truncated(Normal(0.87, 0.12), 0.60, 1.30),
    βᵥ_peak = truncated(Normal(1.0, 0.35), 0.60, 1.30),
    βᵣ_base = truncated(Normal(0.50, 0.10), 0.25, 0.72),
    βᵣ_peak = truncated(Normal(1.05, 0.12), 0.55, 1.10),
    i0c     = truncated(Normal(50.0, 8.0), 5.0, 100.0),    # keep — starting level is correct
    ε_cov   = truncated(Exponential(0.05), 0.0, 0.08),
    ε_flu = truncated(Exponential(1.0), 0.1, 5.0),
    ε_rsv = truncated(Exponential(3.0), 0.5, 10.0)
    )
    _ => error("Province '$name' not found")
    end
end

function find_emergence_week(signal, threshold=1.0)
    # find first week where signal exceeds threshold for 2+ consecutive weeks
    for wk in 1:length(signal)-1
        if signal[wk] > threshold && signal[wk+1] > threshold
            return wk
        end
    end
    return length(signal)
end

function check_M_priors()
    df = read_province("Ontario")
    N, immunity_flu, immunity_cov, immunity_rsv = province_params("Ontario") # initial conditions for the ODE, based on province parameters.
    N = round(Int64, N * unique(df.populationcoverage)[1]) # adjust the effective population
    println("population of Ontario, adjusted: $N")

    W_cov_peak = maximum(skipmissing(df.cov_sg))
    W_flu_peak = maximum(skipmissing(df.flu_sg))
    W_rsv_peak = maximum(skipmissing(df.rsv_sg))
    
    # rough peak prevalence assuming R_eff and SEIR dynamics
    # peak_I/N ≈ S0/N × (1 - 1/Reff) × 0.3  (rough rule of thumb)
    Reff_cov = 1.2; Reff_flu = 1.35; Reff_rsv = 1.5
    peak_frac_cov = (1 - immunity_cov) * (1 - 1/Reff_cov) * 0.3
    peak_frac_flu = (1 - immunity_flu) * (1 - 1/Reff_flu) * 0.3
    peak_frac_rsv = (1 - immunity_rsv) * (1 - 1/Reff_rsv) * 0.3
    
    println("Implied log_M_c: $(round(log(W_cov_peak / peak_frac_cov), digits=2))")
    println("Implied log_M_f: $(round(log(W_flu_peak / peak_frac_flu), digits=2))")
    println("Implied log_M_r: $(round(log(W_rsv_peak / peak_frac_rsv), digits=2))")
end


function sym_defn_ode(du, u, p, t)
    # definitions of SEIR models for each covid, flu, rsv virus 
    # parameters are p = [βᵢ, βᵥ, βᵣ, θ, N] for now (update as needed).
    # the theta is the cross-immunity factor, and N is the total population. 

    # if changing these parameters, change in the analysis function also
    σᵢ = 1/2.0; γᵢ = 1/4.0
    σᵥ = 1/3.0; γᵥ = 1/5.0  
    σᵣ = 1/4.0; γᵣ = 1/5.0

    # v: covid, i: influenza, r: rsv
    Sᵢ, Eᵢ, Iᵢ, Rᵢ, 
    Sᵥ, Eᵥ, Iᵥ, Rᵥ,
    Sᵣ, Eᵣ, Iᵣ, Rᵣ = u

    # tight priors — within ±3 weeks of data-informed center
    # t_school_flu = 113.0 
    # t_school_rsv = 94 #115.0 
    # t_school_cov = 168.0 
    βᵢ_base, βᵢ_peak, βᵥ_base, βᵥ_peak, βᵣ_base, βᵣ_peak, 
        θ, N, ε_flu, ε_cov, ε_rsv, 
        t_school_flu, t_school_cov, t_school_rsv = p
    school_flu = 1.0 / (1.0 + exp(-(t - t_school_flu) / 7.0))
    school_cov = 1.0 / (1.0 + exp(-(t - t_school_cov) / 7.0))
    school_rsv = 1.0 / (1.0 + exp(-(t - t_school_rsv) / 7.0))
    βᵢ = βᵢ_base + (βᵢ_peak - βᵢ_base) * school_flu
    βᵥ = βᵥ_base + (βᵥ_peak - βᵥ_base) * school_cov
    βᵣ = βᵣ_base + (βᵣ_peak - βᵣ_base) * school_rsv

    # calculate the forcce of infection 
    sᵢ = exp(-θ * ((Iᵥ + Iᵣ) / N))
    λᵢ = βᵢ * (Iᵢ / N) * sᵢ

    sᵥ = exp(-θ * ((Iᵢ + Iᵣ) / N))
    λᵥ = βᵥ * (Iᵥ / N) * sᵥ

    sᵣ = exp(-θ * ((Iᵢ + Iᵥ) / N))
    λᵣ = βᵣ * (Iᵣ / N) * sᵣ
    
    #flu 
    du[1] = -λᵢ * Sᵢ - ε_flu
    du[2] = λᵢ * Sᵢ - σᵢ * Eᵢ + ε_flu
    du[3] = σᵢ * Eᵢ - γᵢ * Iᵢ
    du[4] = γᵢ * Iᵢ

    #covid
    du[5] = -λᵥ * Sᵥ - ε_cov
    du[6] = λᵥ * Sᵥ - σᵥ * Eᵥ + ε_cov
    du[7] = σᵥ * Eᵥ - γᵥ * Iᵥ
    du[8] = γᵥ * Iᵥ

    #rsv
    du[9] = -λᵣ * Sᵣ - ε_rsv
    du[10] = λᵣ * Sᵣ - σᵣ * Eᵣ + ε_rsv
    du[11] = σᵣ * Eᵣ - γᵣ * Iᵣ
    du[12] = γᵣ * Iᵣ

    # incidence compartments (cumulative) for each virus (remember to take difference to get daily incidence)
    du[13] = σᵢ * Eᵢ  # influenza incidence
    du[14] = σᵥ * Eᵥ  # covid incidence
    du[15] = σᵣ * Eᵣ  # rsv incidence
    
    return nothing # using in-memory update of du 
end


# convolution which maps infections to wastewater signal, using shedding
function make_kernel(shape, scale, len)
    # generate discretized Gamma shedding kernels
    d = Gamma(shape, scale) 
    pdf_vals = [pdf(d, t) for t in 1:len] # Calculate probability mass for each day 1 to len
    return pdf_vals./ sum(pdf_vals) # Normalize to sum to 1
end

# Define kernels based on literature (Sources: Wu et al., Heijnen et al.)
# Length = 21 days (most shedding happens within 3 weeks)
# we actually made the shape/scale these bayesian but they converged to the same values 
# these are solid values. 
const K_COV = make_kernel(2.5, 1.8, 21) 
const K_FLU = make_kernel(2.5, 1.0, 21) 
const K_RSV = make_kernel(2.0, 2.5, 21) 

function sym_convolve_seir(incidence, kernel)
    len_seq = length(incidence)
    len_ker = length(kernel)
    out = zeros(eltype(incidence), len_seq)
    for t in 1:len_seq
        for k in 1:min(t, len_ker)
            out[t] += incidence[t-k+1] * kernel[k]
        end
    end
    return out
end

# set up a Turing model 
@model function inference_model(obs_ww_flu, obs_ww_cov, obs_ww_rsv, hyperparams, province_params)
    
    # unpack the province parameters 
    (; popsize,
    immunity_flu, # pre-immunity levels
    immunity_cov,
    immunity_rsv, 
    t_school_flu, # number of days for spike in flu/cov/rsv
    t_school_cov,
    t_school_rsv,
    n_windin_flu, # number of days flu/cov/rsv < LOD
    n_windin_cov,
    n_windin_rsv) = province_params
    

    # parameters for the wastewater conversion (informed through a backcalculation)
    log_M_c ~ Normal(8.9, 0.4)
    log_M_f ~ Normal(6.9, 0.4)
    #log_M_r ~ Normal(6.3, 0.4)   # updated for raw data
    log_M_r ~ Normal(7.5, 0.5) 
    M_c = exp(log_M_c)
    M_f = exp(log_M_f)
    M_r = exp(log_M_r)

    # controls the noise in the likelihood of the wastewater signal.
    sigma_cov ~ truncated(Normal(0.5, 0.15), 0.1, 0.9)
    sigma_flu ~ truncated(Normal(0.6, 0.15), 0.2, 1.0)
    sigma_rsv ~ truncated(Normal(0.9, 0.2),  0.1, 1.5)

    # Iᵥ: how many people are currently infectious at week 26 with COVID 
    # Prior is informed by the WW signal magnitude (~25-46 gc/L),
    # We don't know M_c yet so we can't back-calculate exactly,
    # but the wind-in weeks will constrain the joint (Iᵥ, M_c) posterior.
    # Upper bound: implausible that >5% of population infectious simultaneously.
   
    # problem with the rounding: When NUTS runs, Iᵥ is not a Float64
    # it is a ForwardDiff.Dual number carrying derivative information. 
    # The moment you call round(Int64, Iᵥ), you strip all derivative information 
    # and force Turing to abandon gradient-based sampling, 
    # falling back to finite differences or throwing errors. 
    # The sampler becomes orders of magnitude slower.

    # base beta: pre-school, must be low enough to explain near-zero early flu
    # R_eff_base = βᵢ_base × 0.57 / 0.25 — want R_eff < 1 during COVID suppression
    # peak beta: post-school, must be high enough for 96x rise
    # R_eff_peak = βᵢ_peak × 0.57 / 0.25 — want R_eff ~2-3    
    βᵢ_base ~ hyperparams.βᵢ_base
    βᵢ_peak ~ hyperparams.βᵢ_peak
    βᵥ_base ~ hyperparams.βᵥ_base
    βᵥ_peak ~ hyperparams.βᵥ_peak
    βᵣ_base ~ hyperparams.βᵣ_base
    βᵣ_peak ~ hyperparams.βᵣ_peak

    # initial values 
    #i0c ~ truncated(Normal(200.0, 500.0), 10.0, popsize * 0.05) #ontario
    #i0c ~ truncated(Normal(5000.0, 2000.0), 100.0, popsize * 0.10)  #BC
    i0c ~ hyperparams.i0c
    i0i = 1.0
    i0r = 1.0

    # susceptible 
    s0i = (1.0 - immunity_flu) * popsize
    s0c = (1.0 - immunity_cov) * popsize
    s0r = (1.0 - immunity_rsv) * popsize
    
    # recovered compartments
    r0i = immunity_flu * popsize
    r0c = immunity_cov * popsize
    r0r = immunity_rsv * popsize

    initial_cond = [
        s0i - i0i, i0i, 0.0, r0i,   # flu:   S E I R
        s0c - i0c, i0c, 0.0, r0c,   # covid: S E I R
        s0r - i0r, i0r, 0.0, r0r,   # rsv:   S E I R
        0.0, 0.0, 0.0               # incidence
    ]
    
    weeks = length(obs_ww_flu) # or obs_ww_cov, obs_ww_rsv, they should all be the same length.
    days = weeks * 7.0  
    tspan = (0.0, days)

    θ ~ Truncated(Normal(0, 2.0), 0, Inf) 
    ε_flu ~ hyperparams.ε_flu
    ε_cov ~ hyperparams.ε_cov
    ε_rsv ~ hyperparams.ε_rsv

    params = [βᵢ_base, βᵢ_peak, βᵥ_base, βᵥ_peak, βᵣ_base, βᵣ_peak, 
        θ, popsize, ε_flu, ε_cov, ε_rsv,
        t_school_flu, t_school_cov, t_school_rsv]
   
    # solve the ODE 
    prob = ODEProblem(sym_defn_ode, initial_cond, tspan, params)
    _sol = solve(prob, Tsit5(), saveat=1.0)
    sol = reduce(hcat, _sol.u)
    # if _sol.retcode != ReturnCode.Success
    #     Turing.@addlogprob! -Inf
    #     return
    # end

    # get the ODE simulated incidence (this will be in days)
    Iᵢ_sim = make_incidence(sol[13, :])
    Iᵥ_sim = make_incidence(sol[14, :])
    Iᵣ_sim = make_incidence(sol[15, :])

    Wᵢ_sim = sym_convolve_seir(Iᵢ_sim ./ popsize, K_FLU) * M_f
    Wᵥ_sim = sym_convolve_seir(Iᵥ_sim ./ popsize, K_COV) * M_c
    Wᵣ_sim = sym_convolve_seir(Iᵣ_sim ./ popsize, K_RSV) * M_r
    
    lod_ww_flu = 1.0 # fix lod values
    lod_ww_rsv = 1.0

    #likelyhood, but we'll have to go in weeks 
    for w in 1:weeks 
        idx_end = w * 7
        idx_end > length(Wᵢ_sim) && continue # if the ODE solution is shorter than the data, skip the likelihood for those weeks (this can happen in early iterations when the sampler proposes parameters that lead to fast dynamics and the ODE solver fails to reach the end of the time span).

        # all this does is add a floor to the primal value but 
        # autodiff propagates dual numbers independently of this floor 
        # so the dual component can still be negative  
        # that's why we take the max (which is a more standard way to do this in AD) 
        # instead of just adding a small number.
        # mu_i = mean(Wᵢ_sim[idx_end-6 : idx_end]) + 1e-6
        # mu_c = mean(Wᵥ_sim[idx_end-6 : idx_end]) + 1e-6
        # mu_r = mean(Wᵣ_sim[idx_end-6 : idx_end]) + 1e-6
        mu_i = max(mean(Wᵢ_sim[idx_end-6 : idx_end]), 1e-6)
        mu_c = max(mean(Wᵥ_sim[idx_end-6 : idx_end]), 1e-6)
        mu_r = max(mean(Wᵣ_sim[idx_end-6 : idx_end]), 1e-6)
        
        #covid likelihood: 
        if !ismissing(obs_ww_cov[w]) 
            # this is equivalent to 
            # x[i] ~ Normal(m[i]) <--> logpdf(Normal(m[i]), x[i])
            # and then the logpdf is added using the @addlogprob! macro.
            obs_ww_cov[w] ~ LogNormal(log(mu_c), sigma_cov)
        end

        # for flu and rsv, ignore the wind-in period 
        # this is because WW obs during this time is low 
        if w > n_windin_flu    
            if obs_ww_flu[w] <= lod_ww_flu   # at LOD: P(X <= LOD)
                Turing.@addlogprob! logcdf(LogNormal(log(mu_i), sigma_flu), lod_ww_flu)
                #Turing.@addlogprob! logpdf(LogNormal(log(mu_i), sigma_flu * 2.0), lod_ww_flu)
            else
                obs_ww_flu[w] ~ LogNormal(log(mu_i), sigma_flu)
            end
        end
        if w > n_windin_rsv            
            if obs_ww_rsv[w] <= lod_ww_rsv   # at LOD: P(X <= LOD)
                Turing.@addlogprob! logcdf(LogNormal(log(mu_r),  sigma_rsv * 2.0), lod_ww_rsv)
                #Turing.@addlogprob! logpdf(LogNormal(log(mu_r), sigma_rsv * 2.0),  lod_ww_rsv)
            else
                obs_ww_rsv[w] ~ LogNormal(log(mu_r),  sigma_rsv) 
            end
        end
    end
end

function run_inference(name) 
    df, province_params = read_province(name) 

    obs_flu = df.flu_sg
    obs_covid = df.cov_sg
    obs_rsv = df.rsv_sg
    @assert length(df.flu_sg) == length(df.cov_sg) == length(df.rsv_sg)

    hp = hyperparameters(name)

    # for type stability: Turing.DynamicPPL.DebugUtils.model_warntype(_m1)    
    _m1 = inference_model(obs_flu, obs_covid, obs_rsv, hp, province_params)
    #return Turing.DynamicPPL.DebugUtils.model_warntype(_m1)    
    m1 = sample(_m1, NUTS(), MCMCThreads(), 1000, 4)
    #println("finished inference")
    return _m1, m1
end 

## analysis functions 
function calculate_reproduction_number(chain, immunity_flu, immunity_cov, immunity_rsv)
    # R0_bio = β / γ
    # R_eff = R0_bio * (1 - immunity)
    # return R0_bio, R_eff
     # calculating beta -> reproduction numbers (simple estimate) 
    # extract posterior means for base and peak betas
    βᵢ_base = mean(chain[:, :βᵢ_base, :])
    βᵢ_peak = mean(chain[:, :βᵢ_peak, :])
    βᵥ_base = mean(chain[:, :βᵥ_base, :])
    βᵥ_peak = mean(chain[:, :βᵥ_peak, :])
    βᵣ_base = mean(chain[:, :βᵣ_base, :])
    βᵣ_peak = mean(chain[:, :βᵣ_peak, :])

    γᵢ = 1/4.0
    γᵥ = 1/5.0
    γᵣ = 1/5.0 

    # biological R0 = β / γ (in fully susceptible population)
    # effective R = R0_bio × (1 - immunity) = β / γ × S(0)/N

    println("=== Influenza ===")
    println("  Pre-school:  R0_bio=$(round(βᵢ_base/γᵢ, digits=2)),  R_eff=$(round(βᵢ_base/γᵢ*(1-immunity_flu), digits=2))")
    println("  Post-school: R0_bio=$(round(βᵢ_peak/γᵢ, digits=2)),  R_eff=$(round(βᵢ_peak/γᵢ*(1-immunity_flu), digits=2))")
    println("  Expected R_eff range: pre < 1.0 (sub-threshold), post 1.5-2.5")

    println("=== COVID ===")
    println("  Pre-school:  R0_bio=$(round(βᵥ_base/γᵥ, digits=2)),  R_eff=$(round(βᵥ_base/γᵥ*(1-immunity_cov), digits=2))")
    println("  Post-school: R0_bio=$(round(βᵥ_peak/γᵥ, digits=2)),  R_eff=$(round(βᵥ_peak/γᵥ*(1-immunity_cov), digits=2))")
    println("  Expected R_eff range: pre ~1.0-1.2, post ~1.3-1.8")

    println("=== RSV ===")
    println("  Pre-school:  R0_bio=$(round(βᵣ_base/γᵣ, digits=2)),  R_eff=$(round(βᵣ_base/γᵣ*(1-immunity_rsv), digits=2))")
    println("  Post-school: R0_bio=$(round(βᵣ_peak/γᵣ, digits=2)),  R_eff=$(round(βᵣ_peak/γᵣ*(1-immunity_rsv), digits=2))")
    println("  Expected R_eff range: pre ~0.8-1.1, post ~1.5-2.0")
end

function plot_predictive(chain, name)
    # performs the analysis of the posterior samples
    # and plots the predictive distribution of the wastewater signal,
    # as well as the inferred incidence and prevalence curves.
    
    # get the raw data 
    df, province_params = read_province(name)
    # unpack the province parameters 
    (; popsize,
    immunity_flu, # pre-immunity levels
    immunity_cov,
    immunity_rsv, 
    t_school_flu, # number of days for spike in flu/cov/rsv
    t_school_cov,
    t_school_rsv,
    n_windin_flu, # number of days flu/cov/rsv < LOD
    n_windin_cov,
    n_windin_rsv) = province_params
    display(province_params)

    obs_flu = df.flu_sg
    obs_cov = df.cov_sg
    obs_rsv = df.rsv_sg
    obs_flu_raw = df.flu
    obs_cov_raw = df.covN2
    obs_rsv_raw = df.rsv
    
    weeks = length(obs_flu)
    days = weeks * 7 + 1

    # run the ODE model based on posteriors
    nsamples = 1000 
    samples = get(sample(chain, nsamples); section=:parameters)

    # empty matrices to store WW results
    ww_cov_mat = zeros(nsamples, weeks)
    ww_flu_mat = zeros(nsamples, weeks)
    ww_rsv_mat = zeros(nsamples, weeks)

    # empty matrices to store incidence results (in days)
    inc_cov_mat = zeros(nsamples, weeks * 7 + 1) 
    inc_flu_mat = zeros(nsamples, weeks * 7 + 1)
    inc_rsv_mat = zeros(nsamples, weeks * 7 + 1)

    # empty matrices to store prevalnece results (in days)
    prev_cov_mat = zeros(nsamples, weeks * 7 + 1) 
    prev_flu_mat = zeros(nsamples, weeks * 7 + 1)
    prev_rsv_mat = zeros(nsamples, weeks * 7 + 1)

    for i in 1:nsamples
        # extract the parameters for this sample
        θ = samples[:θ][i]
        βᵢ_base = samples[:βᵢ_base][i]
        βᵢ_peak = samples[:βᵢ_peak][i]
        βᵥ_base = samples[:βᵥ_base][i]
        βᵥ_peak = samples[:βᵥ_peak][i]
        βᵣ_base = samples[:βᵣ_base][i]
        βᵣ_peak = samples[:βᵣ_peak][i]
        ε_flu = samples[:ε_flu][i]
        ε_cov = samples[:ε_cov][i]
        ε_rsv = samples[:ε_rsv][i]
        M_c = exp(samples[:log_M_c][i])
        M_f = exp(samples[:log_M_f][i])
        M_r = exp(samples[:log_M_r][i])
        i0c = samples[:i0c][i] 
        i0r = 1
        i0i = 1

        # get initial conditions
        s0i = (1.0 - immunity_flu) * popsize
        s0c = (1.0 - immunity_cov) * popsize
        s0r = (1.0 - immunity_rsv) * popsize
        r0i = immunity_flu * popsize
        r0c = immunity_cov * popsize
        r0r = immunity_rsv * popsize
        u₀ = [
            s0i - i0i, i0i, 0.0, r0i,   # flu:   S E I R
            s0c - i0c, i0c, 0.0, r0c,   # covid: S E I R
            s0r - i0r, i0r, 0.0, r0r,   # rsv:   S E I R
            0.0, 0.0, 0.0               # incidence
        ]
        params = [βᵢ_base, βᵢ_peak, βᵥ_base, βᵥ_peak, βᵣ_base, βᵣ_peak, 
            θ, popsize, ε_flu, ε_cov, ε_rsv, t_school_flu, t_school_cov, t_school_rsv]
        tspan  = (0.0, Float64(weeks * 7))
        prob   = ODEProblem(sym_defn_ode, u₀, tspan, params)
        sol    = solve(prob, Tsit5(), saveat=1.0)
        
        Iᵢ = make_incidence(sol[13, :])
        Iᵥ = make_incidence(sol[14, :])
        Iᵣ = make_incidence(sol[15, :])
        
        
        Wf = sym_convolve_seir(Iᵢ ./ popsize, K_FLU) * M_f
        Wc = sym_convolve_seir(Iᵥ ./ popsize, K_COV) * M_c
        Wr = sym_convolve_seir(Iᵣ ./ popsize, K_RSV) * M_r

        for w in 1:weeks
            idx_end = w * 7
            idx_end > size(Wc, 1) && continue
            ww_cov_mat[i, w] = mean(Wc[idx_end-6:idx_end])
            ww_flu_mat[i, w] = mean(Wf[idx_end-6:idx_end])
            ww_rsv_mat[i, w] = mean(Wr[idx_end-6:idx_end])
        end
        
        # store prevalence results for this sample
        prev_flu_mat[i, :] = sol[3, :]
        prev_cov_mat[i, :] = sol[7, :]
        prev_rsv_mat[i, :] = sol[11, :]

        #store incidence
        inc_flu_mat[i, :] = Iᵢ
        inc_cov_mat[i, :] = Iᵥ
        inc_rsv_mat[i, :] = Iᵣ
    end

    # print statistics
    display(summarystats(chain))  # posteriors
    calculate_reproduction_number(chain, immunity_flu, immunity_cov, immunity_rsv) # effective reproduction number 
    
    # temporal suppression calculation
    _cov = vec(mean(prev_cov_mat, dims = 1))
    _flu = vec(mean(prev_flu_mat, dims = 1))
    _rsv = vec(mean(prev_rsv_mat, dims = 1))
    θ_mean = mean(samples[:θ])
    supp_cov = [1 - exp(-θ_mean * (_flu[t] + _rsv[t])/popsize) for t in 1:days]
    supp_rsv = [1 - exp(-θ_mean * (_flu[t] + _cov[t])/popsize) for t in 1:days]
    supp_flu = [1 - exp(-θ_mean * (_cov[t] + _rsv[t])/popsize) for t in 1:days]
    println("suppression at peak: 
                flu: $(round(maximum(supp_flu), digits=2))
                cov: $(round(maximum(supp_cov), digits=2))
                rsv: $(round(maximum(supp_rsv), digits=2))")
    
    #final size:
    println("final size (in terms of percentage of eff population)
                flu: $(round(sum(vec(mean(inc_flu_mat, dims = 1))) / popsize * 100, digits=1))%
                covid: $(round(sum(vec(mean(inc_cov_mat, dims = 1))) / popsize * 100, digits=1))%
                rsv: $(round(sum(vec(mean(inc_rsv_mat, dims = 1))) / popsize * 100, digits=1))%")
    
    # peak predictions (peak magnitude, peak timing)
    flu_model_peak_max, flu_model_peak_time = findmax(vec(mean(ww_flu_mat, dims = 1)))
    cov_model_peak_max, cov_model_peak_time = findmax(vec(mean(ww_cov_mat, dims = 1)))
    rsv_model_peak_max, rsv_model_peak_time = findmax(vec(mean(ww_rsv_mat, dims = 1)))
    flu_sg_peak_max, flu_sg_peak_time = findmax(obs_flu) 
    cov_sg_peak_max, cov_sg_peak_time = findmax(obs_cov)
    rsv_sg_peak_max, rsv_sg_peak_time = findmax(obs_rsv)
    flu_orig_peak_max, flu_orig_peak_time = findmax(df.flu)
    cov_orig_peak_max, cov_orig_peak_time = findmax(df.covN2)
    rsv_orig_peak_max, rsv_orig_peak_time = findmax(df.rsv)

    # make a matrix with disease in the rows 
    # and times in the columns 1:3 and data in the columns 4:6
    mat = [
        "Flu" flu_model_peak_time flu_sg_peak_time flu_orig_peak_time;
        "Covid" cov_model_peak_time cov_sg_peak_time cov_orig_peak_time;
        "RSV" rsv_model_peak_time rsv_sg_peak_time rsv_orig_peak_time
    ]
    display(pretty_table(mat, backend=:text, column_labels = ["Peak Timing", "Model", "SG", "Raw"]))

    mat = [
        "Flu" flu_model_peak_max flu_sg_peak_max flu_orig_peak_max;
        "Covid" cov_model_peak_max cov_sg_peak_max cov_orig_peak_max;
        "RSV" rsv_model_peak_max rsv_sg_peak_max rsv_orig_peak_max
    ]
    display(pretty_table(mat, backend=:text, column_labels = ["Peak Values", "Model", "SG", "Raw"]))

    #print the WW vectors 
    println("
    flu vector: 
        observed: $(round.(obs_flu, digits=2))
        modelled: $(round.(vec(mean(ww_flu_mat, dims = 1)), digits=2))
    ")
    println("
    cov vector: 
        observed: $(round.(obs_cov, digits=2))
        modelled: $(round.(vec(mean(ww_cov_mat, dims = 1)), digits=2))
    ")
    println("
    rsv vector: 
        observed: $(round.(obs_rsv, digits=2))
        modelled: $(round.(vec(mean(ww_rsv_mat, dims = 1)), digits=2))
    ")
    
    
    # # find the peak week for each individual sample
    # peak_weeks_rsv = [argmax(ww_rsv_mat[i, :]) for i in 1:nsamples]
    # peak_weeks_flu = [argmax(ww_flu_mat[i, :]) for i in 1:nsamples]

    # println("RSV peak week distribution:")
    # println("  Median: $(median(peak_weeks_rsv))")
    # println("  Mean:   $(round(mean(peak_weeks_rsv), digits=1))")
    # println("  Std:    $(round(std(peak_weeks_rsv), digits=1))")
    # println("  2.5%-97.5%: [$(quantile(peak_weeks_rsv, 0.025)), $(quantile(peak_weeks_rsv, 0.975))]")
    # println("  Observed: 29")

    #plot the posteriors and trace plots 
    @gp "reset"
    @gp :- "set multiplot layout 3, 1" :-
    @gp :- 1 1:weeks obs_flu "with lines title 'Obs Flu' lw 3 lc rgb 'black'" :-
    @gp :- 2 1:weeks obs_cov "with lines title 'Obs Covid' lw 3 lc rgb 'black'" :-
    @gp :- 3 1:weeks obs_rsv "with lines title 'Obs RSV' lw 3 lc rgb 'black'" :-
    @gp :- 1 1:weeks vec(mean(ww_flu_mat, dims = 1)) "with lines title 'Pred Flu' lw 2 lc rgb 'blue'" :-
    @gp :- 2 1:weeks vec(mean(ww_cov_mat, dims = 1)) "with lines title 'Pred Covid' lw 2 lc rgb 'blue'" :-
    @gp :- 3 1:weeks vec(mean(ww_rsv_mat, dims = 1)) "with lines title 'Pred RSV' lw 2 lc rgb 'blue'" :-
    @gp :- 1 1:weeks obs_flu_raw "with lines title 'Obs Flu Raw' lw 2 lc rgb 'red'" :-
    @gp :- 2 1:weeks obs_cov_raw "with lines title 'Obs Covid Raw' lw 2 lc rgb 'red'" :-
    @gp :- 3 1:weeks obs_rsv_raw "with lines title 'Obs RSV Raw' lw 2 lc rgb 'red'" :-

    Gnuplot.save("outputs/predictive_wastewater.svg", term="svg enhanced size 800,800 font ', 22' background rgb '#FAFAFA'")
    display(@gp)
    

    @gp "reset"
    @gp :- "set multiplot layout 3, 1" :-
    @gp :- 1 1:size(inc_flu_mat, 2) vec(mean(inc_flu_mat, dims = 1)) "with lines title 'Obs Flu' lw 3 lc rgb 'black'" :-
    @gp :- 2 1:size(inc_cov_mat, 2) vec(mean(inc_cov_mat, dims = 1)) "with lines title 'Obs Covid' lw 3 lc rgb 'black'" :-
    @gp :- 3 1:size(inc_rsv_mat, 2) vec(mean(inc_rsv_mat, dims = 1)) "with lines title 'Obs RSV' lw 3 lc rgb 'black'" :-
    Gnuplot.save("outputs/final_size.svg", term="svg enhanced size 800,800 font ', 22' background rgb '#FAFAFA'")
    display(@gp)

    @gp "reset"
    @gp :- "set multiplot layout 3, 1" :-
    @gp :- 1 1:size(inc_flu_mat, 2) cumsum(vec(mean(inc_flu_mat, dims = 1))) "with lines title 'Obs Flu' lw 3 lc rgb 'black'" :-
    @gp :- 2 1:size(inc_cov_mat, 2) cumsum(vec(mean(inc_cov_mat, dims = 1))) "with lines title 'Obs Covid' lw 3 lc rgb 'black'" :-
    @gp :- 3 1:size(inc_rsv_mat, 2) cumsum(vec(mean(inc_rsv_mat, dims = 1))) "with lines title 'Obs RSV' lw 3 lc rgb 'black'" :-
    Gnuplot.save("outputs/final_size.svg", term="svg enhanced size 800,800 font ', 22' background rgb '#FAFAFA'")
    display(@gp)

    @gp "reset" 
    @gp :- 1:days supp_flu "with lines title 'Flu Suppression' lw 2 lc rgb 'red'" :-
    @gp :- 1:days supp_cov "with lines title 'Covid Suppression' lw 2 lc rgb 'blue'" :-
    @gp :- 1:days supp_rsv "with lines title 'RSV Suppression' lw 2 lc rgb 'green'" :-
    Gnuplot.save("outputs/supressive_temporal.svg", term="svg enhanced size 800,800 font ', 22' background rgb '#FAFAFA'")
    display(@gp)

    # @gp "reset"
    # #@gp :- "set multiplot layout $numchains, 1" :-
    # @gp :-  chain[:, :θ, 1] "with lines title 'Chain 1' lc rgb 'blue'" :-
    # @gp :-  chain[:, :θ, 2] "with lines title 'Chain 2' lc rgb 'red'" :-
    # @gp :-  chain[:, :θ, 3] "with lines title 'Chain 3' lc rgb 'green'" :-
    # @gp :-  chain[:, :θ, 4] "with lines title 'Chain 4' lc rgb 'orange'" :-
    # @gp 

    return nothing
end

function solve_seir_model() 
    _, province_params = read_province("Ontario")     
    # unpack the province parameters 
    (; popsize,
    immunity_flu, # pre-immunity levels
    immunity_cov,
    immunity_rsv, 
    t_school_flu, # number of days for spike in flu/cov/rsv
    t_school_cov,
    t_school_rsv,
    n_windin_flu, # number of days flu/cov/rsv < LOD
    n_windin_cov,
    n_windin_rsv) = province_params

    θ = 0.0
    βᵢ_base = 0.35 
    βᵢ_peak = 1.05 
    βᵥ_base = 0.50 
    βᵥ_peak = 0.87 
    βᵣ_base = 0.50 
    βᵣ_peak = 1.05 
    ε_flu = 3.0
    ε_cov = 0.0
    ε_rsv = 3.0
    
    i0c = 200 
    i0i = 1.0
    i0r = 1.0

    # susceptible 
    s0i = (1.0 - immunity_flu) * popsize
    s0c = (1.0 - immunity_cov) * popsize
    s0r = (1.0 - immunity_rsv) * popsize
    
    # recovered compartments
    r0i = immunity_flu * popsize
    r0c = immunity_cov * popsize
    r0r = immunity_rsv * popsize

    u₀ = [
        s0i - i0i, i0i, 0.0, r0i,   # flu:   S E I R
        s0c - i0c, i0c, 0.0, r0c,   # covid: S E I R
        s0r - i0r, i0r, 0.0, r0r,   # rsv:   S E I R
        0.0, 0.0, 0.0               # incidence
    ]
    
    params = [βᵢ_base, βᵢ_peak, βᵥ_base, βᵥ_peak, βᵣ_base, βᵣ_peak, 
        θ, popsize, ε_flu, ε_cov, ε_rsv, t_school_flu, t_school_cov, t_school_rsv]
    
    tspan  = (0.0, 400.0)
    prob = ODEProblem(sym_defn_ode, u₀, tspan, params)
    sol = solve(prob, Tsit5(), saveat=1.0)
    
    iflu = make_incidence(sol.u[14])
    return iflu
end

# function a12()
#     println("hello world")
# end