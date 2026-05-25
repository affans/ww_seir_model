using DataFrames, CSV, TOML, TidierData
using Random, StatsBase, Distributions
using DifferentialEquations 
using Turing, LogDensityProblems
using MCMCChains, KernelDensity
using SavitzkyGolay # filter
using Gnuplot, PrettyTables, TerminalPager, GarishPrint
using Dates, Parameters, Configurations, Match, JSON
using Serialization, Hyperopt, Printf

Gnuplot.options.term = "sixelgd truecolor enhanced size 900,450 scroll font Arial 12"
# when adding new packages, avoid precompilation by
# pkg> add --preserve=all Example 
# and see https://pkgdocs.julialang.org/dev/environments/#Controlling-Auto-precompilation

function gp_config()
    Gnuplot.options.term = "sixelgd truecolor enhanced size 900,450 scroll font 'Helvetica, 10'"
    # colorscheme from colorbrewer
    return """
    set style line 1 lw 2.5 lt 1 lc rgb '#66c2a5'   # teal
    set style line 2 lw 2.5 lt 1 lc rgb '#fc8d62'   # orange
    set style line 3 lw 2.5 lt 1 lc rgb '#8da0cb'   # blue-purple
    set style line 4 lw 2.5 lt 1 lc rgb '#e78ac3'   # pink
    set style line 5 lw 2.5 lt 1 lc rgb '#a6d854'   # green
    set style line 6 lw 2.5 lt 1 lc rgb '#ffd92f'   # yellow
    set style line 7 lw 2.5 lt 1 lc rgb '#e5c494'   # tan
    set style line 8 lw 2.5 lt 1 lc rgb '#b3b3b3'   # grey
    # set style line 101 lc rgb '#808080' lt 1 lw 2
    set style line 101 lc rgb '#555555' lt 1 lw 2
    set border 3 front ls 101

    set style line 102 lc rgb '#DDDDDD' lt 1 lw 1
    set grid back ls 102
    set tics nomirror out scale 0.75
    set xtics textcolor rgb '#444444'
    set format '%g'
    set samples 2000
    """
end

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
        @filter(EpiYear == 2024 && EpiWeek >= 27 || EpiYear == 2025 && EpiWeek <= 34) 
        @select(pruid, site, city, province, weekstart, EpiYear, EpiWeek, measureid, w_avg, populationcoverage)
        @pivot_wider(df, names_from = measureid, values_from = w_avg)
        @mutate(flu = fluA + fluB) # combine fluA and fluB into one signal. 
        @mutate(cov_sg = ~apply_sg(covN2)) # the ~ means not to vectorize. 
        @mutate(flu_sg = ~apply_sg(flu)) # the ~ means not to vectorize. 
        @mutate(rsv_sg = ~apply_sg(rsv)) # the ~ means not to vectorize.
    end    

    # print the data
    println("adjusted pop size: $(params_obj.popsize)")
    println("=== raw WW data for $name ===")
    println("Flu:  $(round.(df.flu_sg, digits=2))")
    println("Covid: $(round.(df.cov_sg, digits=2))")
    println("RSV:  $(round.(df.rsv_sg, digits=2))")

    return df::DataFrame, params_obj
end

function plot_province(name) 
     # # plot the data
    df, _ = read_province(name)
    @gp "reset" 
    @gp :- gp_config() :-
    @gp :- "set multiplot layout 1, 3" :- # 1 row, 3 columns
    @gp :- 1 1:nrow(df) df.covN2 "with lines notitle lt 0 lc rgb 'black'" :- 
    @gp :- 2 1:nrow(df) df.flu "with lines notitle lt 0 lc rgb 'black'" :- 
    @gp :- 3 1:nrow(df) df.rsv "with lines notitle lt 0 lc rgb 'black'" :- 
    @gp :- 1 1:nrow(df) df.cov_sg "with lines notitle ls 1" :-
    @gp :- 3 1:nrow(df) df.rsv_sg "with lines notitle ls 1" :- 
    @gp :- 2 1:nrow(df) df.flu_sg "with lines notitle ls 1" :-
    #Gnuplot.save("ww_all_prov.svg", term="svg enhanced size 900, 350 font 'Helvetica, 16'")
    println("plot set, call using @gp in terminal to display.") 
end

function hyperparameters(name)
    _, params = read_province(name) 

    # school timing priors are province-specific in center (from TOML)
    # but share the same uncertainty structure across provinces.
    _t_sch_flu = truncated(Normal(100, 14.0), 100 - 28.0, 100 + 28.0)
    _t_sch_cov = truncated(Normal(150, 10.0), 150 - 21.0, 150 + 21.0)
    _t_sch_rsv = truncated(Normal(100, 10.0), 100 - 21.0, 100 + 21.0)

    @match name begin
    "Ontario" => (
    βᵢ_base = truncated(Normal(0.35, 0.10), 0.15, 0.42),
    βᵢ_peak = truncated(Normal(1.05, 0.15), 0.50, 1.50),
    βᵥ_base = truncated(Normal(0.645, 0.1), 0.625, 0.68),
    βᵥ_peak = truncated(Normal(1.10, 0.12), 0.60, 1.30),
    βᵣ_base = truncated(Normal(0.50, 0.10), 0.25, 0.72),
    βᵣ_peak = truncated(Normal(1.05, 0.12), 0.55, 1.10),
    #i0c = truncated(Normal(200.0, 500.0), 10.0, params.popsize * 0.05),
    i0c = truncated(Normal(2000.0, 500.0), 500.0, params.popsize * 0.05),
    ε_flu = truncated(Exponential(1.0), 0.1, 5.0),
    ε_cov = truncated(Exponential(0.05), 0.0, 0.1),
    #ε_cov = truncated(Exponential(1.0), 0.0, 0.1),
    ε_rsv = truncated(Exponential(3.0), 0.5, 10.0),
    t_sch_flu = _t_sch_flu,
    t_sch_cov = _t_sch_cov,
    t_sch_rsv = _t_sch_rsv
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
    i0c = truncated(Normal(2000.0, 800.0), 500.0, params.popsize * 0.10),
    ε_flu = truncated(Exponential(1.0), 0.1, 5.0),
    ε_cov = truncated(Exponential(0.05), 0.0, 0.1),
    ε_rsv = truncated(Exponential(3.0), 0.5, 10.0),
    t_sch_flu = _t_sch_flu,
    t_sch_cov = _t_sch_cov,
    t_sch_rsv = _t_sch_rsv
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
    ε_rsv = truncated(Exponential(3.0), 0.5, 10.0),
    t_sch_flu = _t_sch_flu,
    t_sch_cov = _t_sch_cov,
    t_sch_rsv = _t_sch_rsv
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
    ε_rsv = truncated(Exponential(3.0), 0.5, 10.0),
    t_sch_flu = _t_sch_flu,
    t_sch_cov = _t_sch_cov,
    t_sch_rsv = _t_sch_rsv
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
    ε_rsv = truncated(Exponential(3.0), 0.5, 10.0),
    t_sch_flu = _t_sch_flu,
    t_sch_cov = _t_sch_cov,
    t_sch_rsv = _t_sch_rsv
    )
    "Nova Scotia" => (
    βᵢ_base = truncated(Normal(0.35, 0.10), 0.15, 0.42),
    βᵢ_peak = truncated(Normal(1.05, 0.15), 0.65, 1.50),
    βᵥ_base = truncated(Normal(0.45, 0.08), 0.25, 0.62),   # sub-threshold — COVID declining after week 5
    βᵥ_peak = truncated(Normal(1.0, 0.35), 0.60, 1.30),
    βᵣ_base = truncated(Normal(0.50, 0.10), 0.25, 0.72),
    βᵣ_peak = truncated(Normal(1.05, 0.12), 0.55, 1.10),
    i0c     = truncated(Normal(50.0, 8.0), 5.0, 100.0),    # keep — starting level is correct
    #ε_cov   = truncated(Exponential(0.05), 0.0, 0.08),
    ε_cov   = truncated(Exponential(1.0), 0.0, 0.08),
    ε_flu = truncated(Exponential(1.0), 0.1, 5.0),
    ε_rsv = truncated(Exponential(3.0), 0.5, 10.0),
    t_sch_flu = _t_sch_flu,
    t_sch_cov = _t_sch_cov,
    t_sch_rsv = _t_sch_rsv
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
@model function inference_model(obs_ww_flu, obs_ww_cov, obs_ww_rsv, hyperparams, province_params; nullmodel=false)
    # if nullmodel == true, theta is set to zero manually 
    # meaning there is no sampled theta - adjust post hoc scripts accordingly.

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

    if nullmodel 
        θ = 0.0 
    else 
        θ ~ Truncated(Normal(0, 2.0), 0, Inf) 
    end

    ε_flu ~ hyperparams.ε_flu
    ε_cov ~ hyperparams.ε_cov
    ε_rsv ~ hyperparams.ε_rsv

    t_sch_flu ~ hyperparams.t_sch_flu 
    t_sch_cov ~ hyperparams.t_sch_cov 
    t_sch_rsv ~ hyperparams.t_sch_rsv
    params = [βᵢ_base, βᵢ_peak, βᵥ_base, βᵥ_peak, βᵣ_base, βᵣ_peak, 
        θ, popsize, ε_flu, ε_cov, ε_rsv,
        t_sch_flu, t_sch_cov, t_sch_rsv]
   
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

function run_inference(name, m0_or_m1) 
    m0_or_m1 in (:null, :m1) || error("null_or_alter must be either :null or :m1")
    df, province_params = read_province(name) 

    obs_flu = df.flu_sg
    obs_covid = df.cov_sg
    obs_rsv = df.rsv_sg
    @assert length(df.flu_sg) == length(df.cov_sg) == length(df.rsv_sg)

    hp = hyperparameters(name)

    # for type stability: Turing.DynamicPPL.DebugUtils.model_warntype(_m1)    
    if m0_or_m1 == :m1
        _m1 = inference_model(obs_flu, obs_covid, obs_rsv, hp, province_params; nullmodel=false)
    else 
        _m1 = inference_model(obs_flu, obs_covid, obs_rsv, hp, province_params; nullmodel=true)
    end
    chains = sample(_m1, NUTS(), MCMCThreads(), 1000, 4)    
    return chains
end 

function test_ad() 
    # this function just tested whether Reverse mode or Forward mode AD is faster 
    # Reverse mode only when n parameters > 50, 100 (i.e., large number of parameters)
    # requires Mooncake (reverse diff), SciMLSensitivity, and DynamicPPL.TestUtils.AD
    # Reversediff does not work with ODE - known problem. 
    name = "Ontario"
    df, province_params = read_province(name) 
    obs_flu = df.flu_sg
    obs_covid = df.cov_sg
    obs_rsv = df.rsv_sg
    @assert length(df.flu_sg) == length(df.cov_sg) == length(df.rsv_sg)
    hp = hyperparameters(name)
    _m1 = inference_model(obs_flu, obs_covid, obs_rsv, hp, province_params; nullmodel=false)
    
    # needs using DynamicPPL.TestUtils.AD: run_ad, ADResult
    # using Mooncake, SciMLSensitivity
    println("measuring time") # needs using DynamicPPL.TestUtils.AD: run_ad, ADResult
    for adtype in [AutoForwardDiff(), AutoMooncake()]
        result = run_ad(_m1, adtype; benchmark=true)
        @show result.grad_time / result.primal_time
    end
end

function save_chain(filename, chain)
    # can save as a serialized file
    Serialization.serialize(filename, chain)
end

function read_chain(filename)
    return Serialization.deserialize(filename)
end

function run_all_scenarios() 
    provs = ["Ontario", "British Columbia", "Alberta", "Manitoba", "Quebec", "Nova Scotia"]
    for prov in provs
        m0 = run_inference(prov, :null)
        m1 = run_inference(prov, :m1)
        save_chain("outputs/chain_$(prov)_null.jls", m0)
        save_chain("outputs/chain_$(prov)_m1.jls", m1)
    end
end

function _optimize_ontario(iflu, icov, irsv, tflu, trsv) 
    # objective function for comparing inference results to observed data, 
    # for example by simulating the ODE with posterior mean parameters and calculating a goodness-of-fit metric.
    ### temp function 
    df, _ = read_province("Ontario")
    obs_flu = df.flu_sg
    obs_cov = df.cov_sg
    obs_rsv = df.rsv_sg

    data = TOML.parsefile("parameters.toml")
    province_params = from_dict(ProvinceParams, data["Ontario"]; 
        immunity_flu = iflu,
        immunity_cov = icov,
        immunity_rsv = irsv,
        t_school_flu = tflu,
        t_school_rsv = trsv
    ) # convert Configurations struct
    
    _fname = "iflu" * string(round(Int, iflu * 100)) * "_" * 
                "icov" * string(round(Int, icov * 100)) * "_" * 
                "irsv" * string(round(Int, irsv * 100)) * "_" * 
                "tflu" * string(round(Int, tflu)) * "_" * 
                "trsv" * string(round(Int, trsv))
    fname = "/home/asw850/optim_res/ontario_$(_fname).jls"    
    
    hp = hyperparameters("Ontario")
    _m1 = inference_model(obs_flu, obs_cov, obs_rsv, hp, province_params; nullmodel=false)
    m1 = sample(_m1, NUTS(), 4000)  
    save_chain(fname, m1)
    post_mat = chain_analysis(m1, "Ontario", plot_figs=false)

    # compute the errors from the observed data (this assesses the model fit to the data) 
    flu_error = sum((vec(mean(post_mat.ww_flu, dims = 1)) - obs_flu).^2)
    cov_error = sum((vec(mean(post_mat.ww_cov, dims = 1)) - obs_cov).^2)
    rsv_error = sum((vec(mean(post_mat.ww_rsv, dims = 1)) - obs_rsv).^2)
    return flu_error + cov_error + rsv_error
end

function optimize_ontario()
    try_samples = 50
    println("running try_samples = $try_samples for Ontario optimization...")
    @thyperopt for i = 50,
        sampler = RandomSampler(), # This is default if none provided
        _iflu = [0.42, 0.425, 0.43, 0.435, 0.44, 0.445, 0.45, 0.455],
        _icov = [0.64, 0.645, 0.65, 0.655, 0.66, 0.665, 0.67, 0.675, 0.68],
        _irsv = [0.52, 0.525, 0.53, 0.535, 0.54, 0.545, 0.55, 0.555, 0.56],
        _tflu = [105, 110, 113, 116, 119, 122, 125],
        _trsv = [90, 91, 92, 93, 94, 95, 96, 97, 98]
        println(i, "\t", _iflu, "\t", _icov, "\t", _irsv, "   \t")
        _optimize_ontario(_iflu, _icov, _irsv, _tflu, _trsv)
    end
end

#### POST HOC - ANALYSIS FUNCTIONS
# given an incidence matrix (samples x time), calculate the mean, 95% credible interval for the peak magnitude, and the mean peak timing.
function _calc_stats(inc_mat)
    max_magnitudes, max_times = findmax(inc_mat, dims=2)
    mean_magnitude = round(Int, mean(vec(max_magnitudes)))
    low_magnitude = round(Int, quantile(vec(max_magnitudes), 0.025))
    high_magnitude = round(Int, quantile(vec(max_magnitudes), 0.975))
    mean_time = round(Int, median([e[2] / 7 for e in max_times])) # since e.g. e = CartesianIndex(2, 168)
    low_time = round(Int, quantile(vec([e[2] / 7 for e in max_times]), 0.025)) # since e.g. e = CartesianIndex(2, 168)
    low_time = round(Int, quantile(vec([e[2] / 7 for e in max_times]), 0.025)) # since e.g. e = CartesianIndex(2, 168)
    high_time = round(Int, quantile(vec([e[2] / 7 for e in max_times]), 0.975)) # since e.g. e = CartesianIndex(2, 168)
    return mean_magnitude, low_magnitude, high_magnitude, 
            mean_time, low_time, high_time
end

function _calculate_Rt(t, t_school, beta_base, beta_peak, N, S, P1, P2, θ)
    # internal function to be called from chain_analysis() 
    γᵢ = 1/4.0
    tl = @. logistic((t - t_school)/7) # tl is vector
    βt = @. beta_base + (beta_peak - beta_base) * tl
    S_frac = S / N
    interference =  exp.(-θ .* (P1 .+ P2) ./ N )
    rt =  @. βt / γᵢ * S_frac * interference    
    return rt
end

function chain_analysis(chain, name; plot_figs = true)
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
    n_samples = 1000 
    samples = get(sample(chain, n_samples); section=:parameters)
    
    # empty matrices to store WW results
    ww_cov = zeros(n_samples, weeks)
    ww_flu = zeros(n_samples, weeks)
    ww_rsv = zeros(n_samples, weeks)

    # empty matrices to store incidence results (in days)
    inc_cov = zeros(n_samples, weeks * 7 + 1) 
    inc_flu = zeros(n_samples, weeks * 7 + 1)
    inc_rsv = zeros(n_samples, weeks * 7 + 1)

    # empty matrices to store prevalnece results (in days)
    prev_cov = zeros(n_samples, weeks * 7 + 1) 
    prev_flu = zeros(n_samples, weeks * 7 + 1)
    prev_rsv = zeros(n_samples, weeks * 7 + 1)

    # effective reproduction number
    Rt_flu = zeros(n_samples, weeks * 7 + 1)
    Rt_cov = zeros(n_samples, weeks * 7 + 1)
    Rt_rsv = zeros(n_samples, weeks * 7 + 1)

    # counterfactual (i.e., theta = 0 but not the NULL model)
    Rt_flu_cf = zeros(n_samples, weeks * 7 + 1)
    Rt_cov_cf = zeros(n_samples, weeks * 7 + 1)
    Rt_rsv_cf = zeros(n_samples, weeks * 7 + 1)

    # see if theta is in the model (it won't be in the null model)
    mtype = Symbol("θ") in keys(chain)
    mtype ? println("Model includes θ (cross-immunity factor)") : println("Model is the NULL, does NOT include θ (null model)")
    
    for i in 1:n_samples
        # extract the parameters for this sample
        θ = mtype ? samples[:θ][i] : 0.0
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
        t_school_cov = samples[:t_sch_cov][i]
        t_school_flu = samples[:t_sch_flu][i]
        t_school_rsv = samples[:t_sch_rsv][i]

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
    
        Sᵢ = sol[1, :]
        Sᵥ = sol[5, :]
        Sᵣ = sol[9, :]

        Pᵢ = sol[3, :]
        Pᵥ = sol[7, :]
        Pᵣ = sol[11, :]

        Iᵢ = make_incidence(sol[13, :])
        Iᵥ = make_incidence(sol[14, :])
        Iᵣ = make_incidence(sol[15, :])
                
        Wf = sym_convolve_seir(Iᵢ ./ popsize, K_FLU) * M_f
        Wc = sym_convolve_seir(Iᵥ ./ popsize, K_COV) * M_c
        Wr = sym_convolve_seir(Iᵣ ./ popsize, K_RSV) * M_r

        for w in 1:weeks
            idx_end = w * 7
            idx_end > size(Wc, 1) && continue
            ww_cov[i, w] = mean(Wc[idx_end-6:idx_end])
            ww_flu[i, w] = mean(Wf[idx_end-6:idx_end])
            ww_rsv[i, w] = mean(Wr[idx_end-6:idx_end])
        end

        # store prevalence results for this sample
        prev_flu[i, :] = Pᵢ
        prev_cov[i, :] = Pᵥ
        prev_rsv[i, :] = Pᵣ
  
        #store incidence
        inc_flu[i, :] = Iᵢ
        inc_cov[i, :] = Iᵥ
        inc_rsv[i, :] = Iᵣ

        # calculate Rt (make sure to send correct prevalence for each virus (so cov, rsv for flu etc))
        Rt_flu[i, :] = _calculate_Rt(sol.t, t_school_flu, βᵢ_base, βᵢ_peak, popsize, Sᵢ, Pᵥ, Pᵣ, θ)
        Rt_cov[i, :] = _calculate_Rt(sol.t, t_school_cov, βᵥ_base, βᵥ_peak, popsize, Sᵥ, Pᵢ, Pᵣ, θ)
        Rt_rsv[i, :] = _calculate_Rt(sol.t, t_school_rsv, βᵣ_base, βᵣ_peak, popsize, Sᵣ, Pᵢ, Pᵥ, θ)

        # calculate Rt counterfactual (theta = 0, but not NULL model so other parameters are the same) 
        Rt_flu_cf[i, :] = _calculate_Rt(sol.t, t_school_flu, βᵢ_base, βᵢ_peak, popsize, Sᵢ, Pᵥ, Pᵣ, 0)
        Rt_cov_cf[i, :] = _calculate_Rt(sol.t, t_school_cov, βᵥ_base, βᵥ_peak, popsize, Sᵥ, Pᵢ, Pᵣ, 0)
        Rt_rsv_cf[i, :] = _calculate_Rt(sol.t, t_school_rsv, βᵣ_base, βᵣ_peak, popsize, Sᵣ, Pᵢ, Pᵥ, 0)
    end

    # print statistics
    describe(chain)  # posteriors

    # print the main theta and suppression values value   
    if mtype  
        meanθ, lowθ, highθ = round.((mean(vec(samples[:θ])), quantile(vec(samples[:θ]), 0.025), quantile(vec(samples[:θ]), 0.975)), digits=2)
    else 
        meanθ, lowθ, highθ = 0.0, 0.0, 0.0
    end

    # calculate the suppression
    _cov = vec(mean(prev_cov, dims = 1))
    _flu = vec(mean(prev_flu, dims = 1))
    _rsv = vec(mean(prev_rsv, dims = 1))
    supp_cov_mean = round(maximum([1 - exp(-meanθ * (_flu[t] + _rsv[t])/popsize) for t in 1:days]), digits=2)
    supp_rsv_mean = round(maximum([1 - exp(-meanθ * (_flu[t] + _cov[t])/popsize) for t in 1:days]), digits=2)
    supp_flu_mean = round(maximum([1 - exp(-meanθ * (_cov[t] + _rsv[t])/popsize) for t in 1:days]), digits=2)
    supp_cov_low = round(maximum([1 - exp(-lowθ * (_flu[t] + _rsv[t])/popsize) for t in 1:days]), digits=2)
    supp_rsv_low = round(maximum([1 - exp(-lowθ * (_flu[t] + _cov[t])/popsize) for t in 1:days]), digits=2)
    supp_flu_low = round(maximum([1 - exp(-lowθ * (_cov[t] + _rsv[t])/popsize) for t in 1:days]), digits=2)
    supp_cov_high = round(maximum([1 - exp(-highθ * (_flu[t] + _rsv[t])/popsize) for t in 1:days]), digits=2)
    supp_rsv_high = round(maximum([1 - exp(-highθ * (_flu[t] + _cov[t])/popsize) for t in 1:days]), digits=2)
    supp_flu_high = round(maximum([1 - exp(-highθ * (_cov[t] + _rsv[t])/popsize) for t in 1:days]), digits=2)
    println("\nTheta value (and Cr I) θ: $meanθ (95% Cr I: $lowθ, $highθ)")
    println("\nSuppression at peak: 
    Flu: $(supp_flu_mean) (95% Cr I: $supp_flu_low, $supp_flu_high)
    Covid: $(supp_cov_mean) (95% Cr I: $supp_cov_low, $supp_cov_high)
    RSV: $(supp_rsv_mean) (95% Cr I: $supp_rsv_low, $supp_rsv_high)")    
    
    # calculate the Rt means 
    # per-draw peak Rt
    peak_flu     = [maximum(r) for r in eachrow(Rt_flu)]
    peak_cov     = [maximum(r) for r in eachrow(Rt_cov)]
    peak_rsv     = [maximum(r) for r in eachrow(Rt_rsv)]
    peak_flu_cf  = [maximum(r) for r in eachrow(Rt_flu_cf)]
    peak_cov_cf  = [maximum(r) for r in eachrow(Rt_cov_cf)]
    peak_rsv_cf  = [maximum(r) for r in eachrow(Rt_rsv_cf)]
    Rt_flu_mean = round(mean(peak_flu), digits=2)
    Rt_flu_low = round(quantile(peak_flu, 0.05), digits=2)
    Rt_flu_high = round(quantile(peak_flu, 0.95), digits=2)
    Rt_cov_mean = round(mean(peak_cov), digits=2)
    Rt_cov_low = round(quantile(peak_cov, 0.05), digits=2)
    Rt_cov_high = round(quantile(peak_cov, 0.95), digits=2)
    Rt_rsv_mean = round(mean(peak_rsv), digits=2)
    Rt_rsv_low = round(quantile(peak_rsv, 0.05), digits=2)
    Rt_rsv_high = round(quantile(peak_rsv, 0.95), digits=2) 
    # counterfactual Rt Means
    Rt_cf_flu_mean = round(mean(peak_flu_cf), digits=2)
    Rt_cf_flu_low = round(quantile(peak_flu_cf, 0.05), digits=2)
    Rt_cf_flu_high = round(quantile(peak_flu_cf, 0.95), digits=2)
    Rt_cf_cov_mean = round(mean(peak_cov_cf), digits=2)
    Rt_cf_cov_low = round(quantile(peak_cov_cf, 0.05), digits=2)
    Rt_cf_cov_high = round(quantile(peak_cov_cf, 0.95), digits=2)
    Rt_cf_rsv_mean = round(mean(peak_rsv_cf), digits=2)
    Rt_cf_rsv_low = round(quantile(peak_rsv_cf, 0.05), digits=2)
    Rt_cf_rsv_high = round(quantile(peak_rsv_cf, 0.95), digits=2)
    println("Flu:   CF Rt=$(Rt_cf_flu_mean), realised Rt=$(Rt_flu_mean), ")
    println("Covid: CF Rt=$(Rt_cf_cov_mean), realised Rt=$(Rt_cov_mean), ")
    println("RSV:   CF Rt=$(Rt_cf_rsv_mean), realised Rt=$(Rt_rsv_mean), ")

    # rope suppression
    if mtype
        rope_flu = rope_suppression(samples, prev_cov, prev_rsv, popsize)
        rope_cov = rope_suppression(samples, prev_flu, prev_rsv, popsize)
        rope_rsv = rope_suppression(samples, prev_flu, prev_cov, popsize)
        println("\nROPE suppression (proportion of posterior distribution where suppression < 10%):
        Flu: $(round(rope_flu, digits=2))
        Covid: $(round(rope_cov, digits=2))
        RSV: $(round(rope_rsv, digits=2))")
    end
    
    # Plot WW curves (not needed for the paper, but useful for checking model fit and visualizing the results)
    println("=== Wastewater signal (modelled vs observed) ===")
      #print the WW vectors 
    println("Flu vector: 
    observed: $(round.(obs_flu, digits=2))
    modelled: $(round.(vec(mean(ww_flu, dims = 1)), digits=2))
    ")
    println("Covid vector: 
    observed: $(round.(obs_cov, digits=2))
    modelled: $(round.(vec(mean(ww_cov, dims = 1)), digits=2))
    ")
    println("RSV vector: 
    observed: $(round.(obs_rsv, digits=2))
    modelled: $(round.(vec(mean(ww_rsv, dims = 1)), digits=2))
    ")

    # Plot the posterior predictictions of the WW (this assesses the model fit to the data)
    flu_model_peak_max, flu_model_peak_time = findmax(vec(mean(ww_flu, dims = 1)))
    cov_model_peak_max, cov_model_peak_time = findmax(vec(mean(ww_cov, dims = 1)))
    rsv_model_peak_max, rsv_model_peak_time = findmax(vec(mean(ww_rsv, dims = 1)))
    flu_sg_peak_max, flu_sg_peak_time = findmax(obs_flu) 
    cov_sg_peak_max, cov_sg_peak_time = findmax(obs_cov)
    rsv_sg_peak_max, rsv_sg_peak_time = findmax(obs_rsv)
    flu_orig_peak_max, flu_orig_peak_time = findmax(df.flu)
    cov_orig_peak_max, cov_orig_peak_time = findmax(df.covN2)
    rsv_orig_peak_max, rsv_orig_peak_time = findmax(df.rsv)
    # Compute the errors from the observed data (this assesses the model fit to the data) 
    flu_error = sum((vec(mean(ww_flu, dims = 1)) - obs_flu).^2)
    cov_error = sum((vec(mean(ww_cov, dims = 1)) - obs_cov).^2)
    rsv_error = sum((vec(mean(ww_rsv, dims = 1)) - obs_rsv).^2)
    
    # Calculate the times and magnitude at the peaks
    mat = [
        "Flu" flu_model_peak_time flu_sg_peak_time flu_orig_peak_time flu_model_peak_max flu_sg_peak_max flu_orig_peak_max flu_error;
        "Covid" cov_model_peak_time cov_sg_peak_time cov_orig_peak_time cov_model_peak_max cov_sg_peak_max cov_orig_peak_max cov_error;
        "RSV" rsv_model_peak_time rsv_sg_peak_time rsv_orig_peak_time rsv_model_peak_max rsv_sg_peak_max rsv_orig_peak_max rsv_error
    ]
    println("Peak magnitude and timing of WW signal (modelled vs observed, and error from observed data)")
    pretty_table(mat, backend=:text, 
        column_labels = [["", "Model", "SG", "Raw", "Model", "SG", "Raw", "∑Error"],["", "", "", "", "", "", "", ""]],
        merge_column_label_cells = [MergeCells(2, 2, 3, "Peak Time", :c), MergeCells(2, 5, 3, "Peak Magnitude", :c)])

    ## Print peak and timing of incidence curves
    inc_peak_matrix = Matrix{Any}(nothing, 0, 3) # 
    for (inc_matrix, disease) in zip((inc_flu, inc_cov, inc_rsv), ("Flu", "Covid", "RSV"))
        m, lm, hm, tm, ltm, htm = _calc_stats(inc_matrix)
        inc_peak_matrix = vcat(inc_peak_matrix, [string(disease) "$m ($lm, $hm)" "$tm ($ltm, $htm)"])
    end
    println("Peak magnitude and timing of incidence curves (mean and 95% credible interval)")
    pretty_table(inc_peak_matrix, backend=:text, column_labels = ["Disease", "Peak Magnitude (95% CI)", "Peak Timing (95% CI)"])

    if plot_figs
        #plot the posteriors and trace plots 
        @gp "reset"
        @gp :- "set multiplot layout 3, 1" :-
        @gp :- 1 1:weeks obs_flu "with lines title 'Obs Flu' lw 3 lc rgb 'black'" :-
        @gp :- 2 1:weeks obs_cov "with lines title 'Obs Covid' lw 3 lc rgb 'black'" :-
        @gp :- 3 1:weeks obs_rsv "with lines title 'Obs RSV' lw 3 lc rgb 'black'" :-
        @gp :- 1 1:weeks vec(mean(ww_flu, dims = 1)) "with lines title 'Pred Flu' lw 2 lc rgb 'blue'" :-
        @gp :- 2 1:weeks vec(mean(ww_cov, dims = 1)) "with lines title 'Pred Covid' lw 2 lc rgb 'blue'" :-
        @gp :- 3 1:weeks vec(mean(ww_rsv, dims = 1)) "with lines title 'Pred RSV' lw 2 lc rgb 'blue'" :-
        @gp :- 1 1:weeks obs_flu_raw "with lines title 'Obs Flu Raw' lw 2 lc rgb 'red'" :-
        @gp :- 2 1:weeks obs_cov_raw "with lines title 'Obs Covid Raw' lw 2 lc rgb 'red'" :-
        @gp :- 3 1:weeks obs_rsv_raw "with lines title 'Obs RSV Raw' lw 2 lc rgb 'red'" :-
        #Gnuplot.save("outputs/predictive_wastewater.svg", term="svg enhanced size 800,800 font ', 22' background rgb '#FAFAFA'")
        display(@gp)

        @gp "reset"
        @gp :- "set multiplot layout 3, 1" :-
        @gp :- 1 1:size(inc_flu, 2) vec(mean(inc_flu, dims = 1)) "with lines title 'Obs Flu' lw 3 lc rgb 'black'" :-
        @gp :- 2 1:size(inc_cov, 2) vec(mean(inc_cov, dims = 1)) "with lines title 'Obs Covid' lw 3 lc rgb 'black'" :-
        @gp :- 3 1:size(inc_rsv, 2) vec(mean(inc_rsv, dims = 1)) "with lines title 'Obs RSV' lw 3 lc rgb 'black'" :-
        #Gnuplot.save("outputs/final_size.svg", term="svg enhanced size 800,800 font ', 22' background rgb '#FAFAFA'")
        display(@gp)

        @gp "reset"
        @gp :- "set multiplot layout 3, 1" :-
        @gp :- 1 1:size(inc_flu, 2) cumsum(vec(mean(inc_flu, dims = 1))) "with lines title 'Obs Flu' lw 3 lc rgb 'black'" :-
        @gp :- 2 1:size(inc_cov, 2) cumsum(vec(mean(inc_cov, dims = 1))) "with lines title 'Obs Covid' lw 3 lc rgb 'black'" :-
        @gp :- 3 1:size(inc_rsv, 2) cumsum(vec(mean(inc_rsv, dims = 1))) "with lines title 'Obs RSV' lw 3 lc rgb 'black'" :-
        #Gnuplot.save("outputs/final_size.svg", term="svg enhanced size 800,800 font ', 22' background rgb '#FAFAFA'")
        display(@gp)
    end 
    # return the posterior predictive matrices for further analysis if needed
    return (; inc_flu, inc_cov, inc_rsv,
            prev_flu, prev_cov, prev_rsv,
            ww_flu, ww_cov, ww_rsv, 
            meanθ, lowθ, highθ,
            supp_flu_mean, supp_cov_mean, supp_rsv_mean,
            supp_flu_low, supp_cov_low, supp_rsv_low,
            supp_flu_high, supp_cov_high, supp_rsv_high, Rt_flu, Rt_cov, Rt_rsv)
end

function rope_suppression(samples, prev_a, prev_b, popsize; threshold=0.10)
    n_draws = length(samples[:θ])
    mask = ones(Bool, n_draws)
    p_above = 0.0
    for s in 1:n_draws
        θ_s = samples[:θ][s]
        # peak combined prevalence across time for this draw
        peak_combined = maximum((prev_a[s,:] .+ prev_b[s,:]) ./ popsize)
        supp_s = 1 - exp(-θ_s * peak_combined)
        p_above += (supp_s > threshold)
    end
    return p_above / n_draws
end

function run_analysis_all_prov() 
    # runs the chain analysis for all provinces and saves the results to a log file (this is useful for checking the results and for generating summary tables of the results across provinces)
    provs = ["British Columbia", "Alberta", "Ontario", "Manitoba", "Nova Scotia","Quebec"]
    open("outputs/full_results.log", "w") do io
        redirect_stdout(io) do
            for prov in provs
                println("=== Province: $prov === M1 RESULTS ===")
                m1 = read_chain("outputs/chain_$(prov)_m1.jls")
                chain_analysis(m1, prov, plot_figs = false)
                println("=== Province: $prov === M0 / NULL RESULTS ===")
                m0 = read_chain("outputs/chain_$(prov)_null.jls")
                chain_analysis(m0, prov, plot_figs = false)    
            end
        end
    end
end

function plot_wastewater()
    # plots the raw WW signal
    provs = ["Alberta", "Ontario", "Manitoba", "Nova Scotia","Quebec", "British Columbia"]
    @gp "reset" 
    @gp :- gp_config() :-
    @gp :- "set multiplot layout 6, 3" :-
    @gp :- "set label 101 'Alberta' at screen 0.075, 1.00-0.015 left font 'Helvetica,12' tc rgb '#444444'" :-
    @gp :- "set label 102 'Ontario' at screen 0.075, 0.8333-0.015 left font 'Helvetica,12' tc rgb '#444444'" :-
    @gp :- "set label 103 'Manitoba' at screen 0.075, 0.6667-0.015 left font 'Helvetica,12' tc rgb '#444444'" :-
    @gp :- "set label 104 'Nova Scotia' at screen 0.075, 0.5000-0.015 left font 'Helvetica,12' tc rgb '#444444'" :-
    @gp :- "set label 105 'Quebec' at screen 0.075, 0.3333-0.015 left font 'Helvetica,12' tc rgb '#444444'" :-
    @gp :- "set label 106 'British Columbia' at screen 0.075, 0.1667-0.015 left font 'Helvetica,12' tc rgb '#444444'" :-
    @gp :- "set tmargin 2" :-
    for (prov, i) in zip(provs, 1:3:18)
        println(i)
        df, _ = read_province(prov)
        m1 = read_chain("outputs/chain_$(prov)_m1.jls")
        chain = chain_analysis(m1, prov, plot_figs = false)
        obs_flu = df.flu
        obs_cov = df.covN2
        obs_rsv = df.rsv
        #@gp :- "set nokey" :-
        @gp :- i 1:length(obs_flu) obs_flu "with points notitle lw 1 ps 0.2 pt 6 lc rgb 'black'" :-
        @gp :- (i+1) 1:length(obs_cov) obs_cov "with points notitle lw 1 ps 0.2 pt 6 lc rgb 'black'" :-
        @gp :- (i+2) 1:length(obs_rsv) obs_rsv "with points notitle lw 1 ps 0.2 pt 6 lc rgb 'black'" :-
        @gp :- i 1:length(obs_flu) df.flu_sg "smooth csplines with lines notitle  ls 1" :-
        @gp :- (i+1) 1:length(obs_cov) df.cov_sg "smooth csplines with lines notitle ls 2" :-
        @gp :- (i+ 2) 1:length(obs_rsv) df.rsv_sg "smooth csplines with lines notitle ls 3" :- 
    end
    Gnuplot.save("outputs/figures/wastewater_raw_allprovs.pdf", term="pdfcairo enhanced size 6in,9in font 'Helvetica, 12'")
    
    # do another figure with just two provinces for presentation purposes
    # so don't have to deal with indices
    # I changed the color to grey and black here
    @gp "reset" 
    @gp :- gp_config() :-
    @gp :- "set multiplot layout 2, 3" :-
    @gp :- "set label 101 'Alberta' at screen 0.015, 1.00-0.020 left font 'Helvetica Bold,12' tc rgb '#444444'" :-
    @gp :- "set label 102 'Ontario' at screen 0.015, 0.50-0.020 left font 'Helvetica Bold,12' tc rgb '#444444'" :-
    @gp :- "set tmargin 2" :-
    for (prov, i) in zip(["Alberta", "Ontario"], 1:3:6)
        println(i)
        df, _ = read_province(prov)
        m1 = read_chain("outputs/chain_$(prov)_m1.jls")
        chain = chain_analysis(m1, prov, plot_figs = false)
        obs_flu = df.flu
        obs_cov = df.covN2
        obs_rsv = df.rsv
        #@gp :- "set nokey" :-
        @gp :- i 1:length(obs_flu) obs_flu "with points notitle lw 1 ps 0.2 pt 6 lc rgb 'black'" :-
        @gp :- (i+1) 1:length(obs_cov) obs_cov "with points notitle lw 1 ps 0.2 pt 6 lc rgb 'black'" :-
        @gp :- (i+2) 1:length(obs_rsv) obs_rsv "with points notitle lw 1 ps 0.2 pt 6 lc rgb 'black'" :-
        @gp :- i 1:length(obs_flu) df.flu_sg "smooth csplines with lines notitle  ls 1 lc rgb '#bdbdbd'" :-
        @gp :- (i+1) 1:length(obs_cov) df.cov_sg "smooth csplines with lines notitle ls 2 lc rgb '#bdbdbd'" :-
        @gp :- (i+ 2) 1:length(obs_rsv) df.rsv_sg "smooth csplines with lines notitle ls 3 lc rgb '#bdbdbd'" :- 
    end
    Gnuplot.save("outputs/figures/wastewater_raw_twoprovs.pdf", term="pdfcairo enhanced size 6in,3in font 'Helvetica, 12'")
    @gp 
end

function plot_fitted_wastewater_signal() 
    # sampling from the posterior, run the ODE forward and plot the WW
    # panel of 6 (3 x 3), with the observed wastewater signal for flu, covid, and rsv across the 6 provinces.
    provs = ["Alberta", "Ontario", "Manitoba", "Nova Scotia","Quebec", "British Columbia"]
    @gp "reset"
    @gp :- gp_config() :-
    @gp :- "set multiplot layout 6, 3" :-
    @gp :- "set label 101 'Alberta' at screen 0.075, 1.00-0.015 left font 'Helvetica,12' tc rgb '#444444'" :-
    @gp :- "set label 102 'Ontario' at screen 0.075, 0.8333-0.015 left font 'Helvetica,12' tc rgb '#444444'" :-
    @gp :- "set label 103 'Manitoba' at screen 0.075, 0.6667-0.015 left font 'Helvetica,12' tc rgb '#444444'" :-
    @gp :- "set label 104 'Nova Scotia' at screen 0.075, 0.5000-0.015 left font 'Helvetica,12' tc rgb '#444444'" :-
    @gp :- "set label 105 'Quebec' at screen 0.075, 0.3333-0.015 left font 'Helvetica,12' tc rgb '#444444'" :-
    @gp :- "set label 106 'British Columbia' at screen 0.075, 0.1667-0.015 left font 'Helvetica,12' tc rgb '#444444'" :-
    @gp :- "set tmargin 2" :-
    @gp :- "set style fill transparent solid 0.3 noborder" :-
    for (prov, i) in zip(provs, 1:3:18)
        println(i)
        df, _ = read_province(prov)
        m1 = read_chain("outputs/chain_$(prov)_m1.jls")
        chain = chain_analysis(m1, prov, plot_figs = false)
        obs_flu = df.flu
        obs_cov = df.covN2
        obs_rsv = df.rsv
        #@gp :- "set nokey" :-
        @gp :- i 1:length(obs_flu) obs_flu "with points notitle lw 1 ps 0.2 pt 6 lc rgb 'black'" :-
        @gp :- (i+1) 1:length(obs_cov) obs_cov "with points notitle lw 1 ps 0.2 pt 6 lc rgb 'black'" :-
        @gp :- (i+2) 1:length(obs_rsv) obs_rsv "with points notitle lw 1 ps 0.2 pt 6 lc rgb 'black'" :-
        @gp :- i 1:length(obs_flu) vec(mean(chain.ww_flu, dims = 1)) "smooth csplines with lines notitle  ls 1" :-
        @gp :- (i+1) 1:length(obs_cov) vec(mean(chain.ww_cov, dims = 1)) "smooth csplines with lines notitle ls 2" :-
        @gp :- (i+ 2) 1:length(obs_rsv) vec(mean(chain.ww_rsv, dims = 1)) "smooth csplines with lines notitle ls 3" :- 

        
        lo = [quantile(c, 0.05) for c in eachcol(chain.ww_flu)]
        hi = [quantile(c, 0.95) for c in eachcol(chain.ww_flu)]
        @gp :- i 1:length(obs_flu) lo hi "with filledcurves ls 1 notitle" :-

        lo = [quantile(c, 0.05) for c in eachcol(chain.ww_cov)]
        hi = [quantile(c, 0.95) for c in eachcol(chain.ww_cov)]
        @gp :- (i+1) 1:length(obs_cov) lo hi "with filledcurves notitle ls 2" :-

        lo = [quantile(c, 0.05) for c in eachcol(chain.ww_rsv)]
        hi = [quantile(c, 0.95) for c in eachcol(chain.ww_rsv)]
        @gp :- (i+2) 1:length(obs_rsv) lo hi "with filledcurves notitle ls 3" :-
        #@gp :- i 1:length(obs_flu) vec(mean(chain_null.ww_flu, dims = 1)) "smooth csplines with lines title 'M0 Flu' ls 4" :-
        #@gp :- (i+1) 1:length(obs_cov) vec(mean(chain_null.ww_cov, dims = 1)) "smooth csplines with lines title 'M0 Covid' ls 5" :-
        #@gp :- (i+ 2) 1:length(obs_rsv) vec(mean(chain_null.ww_rsv, dims = 1)) "smooth csplines with lines title 'M0 RSV' ls 6" :- 
    end
    Gnuplot.save("outputs/figures/wastewater_fits.pdf", term="pdfcairo enhanced size 6in,9in font 'Helvetica, 12'")
    @gp 

    ## do the same for top three (for presentation purposes)
    @gp "reset"
    @gp :- gp_config() :-
    @gp :- "set multiplot layout 3, 3" :-
    @gp :- "set label 101 'Alberta' at screen 0.015, 1.00-0.015 left font 'Helvetica,12' tc rgb '#444444'" :-
    @gp :- "set label 102 'Ontario' at screen 0.015, 0.66-0.015 left font 'Helvetica,12' tc rgb '#444444'" :-
    @gp :- "set label 103 'Manitoba' at screen 0.015, 0.33-0.015 left font 'Helvetica,12' tc rgb '#444444'" :-
    @gp :- "set tmargin 2" :-
    @gp :- "set style fill transparent solid 0.3 noborder" :- 
    for (prov, i) in zip(("Alberta", "Ontario", "Manitoba"), 1:3:18)
        println(i)
        df, _ = read_province(prov)
        m1 = read_chain("outputs/chain_$(prov)_m1.jls")
        chain = chain_analysis(m1, prov, plot_figs = false)
        obs_flu = df.flu
        obs_cov = df.covN2
        obs_rsv = df.rsv
        #@gp :- "set nokey" :-
        @gp :- i 1:length(obs_flu) obs_flu "with points notitle lw 1 ps 0.2 pt 6 lc rgb 'black'" :-
        @gp :- (i+1) 1:length(obs_cov) obs_cov "with points notitle lw 1 ps 0.2 pt 6 lc rgb 'black'" :-
        @gp :- (i+2) 1:length(obs_rsv) obs_rsv "with points notitle lw 1 ps 0.2 pt 6 lc rgb 'black'" :-
        @gp :- i 1:length(obs_flu) vec(mean(chain.ww_flu, dims = 1)) "smooth csplines with lines notitle  ls 1" :-
        @gp :- (i+1) 1:length(obs_cov) vec(mean(chain.ww_cov, dims = 1)) "smooth csplines with lines notitle ls 2" :-
        @gp :- (i+ 2) 1:length(obs_rsv) vec(mean(chain.ww_rsv, dims = 1)) "smooth csplines with lines notitle ls 3" :- 

        lo = [quantile(c, 0.05) for c in eachcol(chain.ww_flu)]
        hi = [quantile(c, 0.95) for c in eachcol(chain.ww_flu)]
        @gp :- i 1:length(obs_flu) lo hi "with filledcurves ls 1 notitle" :-

        lo = [quantile(c, 0.05) for c in eachcol(chain.ww_cov)]
        hi = [quantile(c, 0.95) for c in eachcol(chain.ww_cov)]
        @gp :- (i+1) 1:length(obs_cov) lo hi "with filledcurves notitle ls 2" :-

        lo = [quantile(c, 0.05) for c in eachcol(chain.ww_rsv)]
        hi = [quantile(c, 0.95) for c in eachcol(chain.ww_rsv)]
        @gp :- (i+2) 1:length(obs_rsv) lo hi "with filledcurves notitle ls 3" :-
        #@gp :- i 1:length(obs_flu) vec(mean(chain_null.ww_flu, dims = 1)) "smooth csplines with lines title 'M0 Flu' ls 4" :-
        #@gp :- (i+1) 1:length(obs_cov) vec(mean(chain_null.ww_cov, dims = 1)) "smooth csplines with lines title 'M0 Covid' ls 5" :-
        #@gp :- (i+ 2) 1:length(obs_rsv) vec(mean(chain_null.ww_rsv, dims = 1)) "smooth csplines with lines title 'M0 RSV' ls 6" :- 
    end
    Gnuplot.save("outputs/figures/wastewater_fits_first3.pdf", term="pdfcairo enhanced size 6in,4.5in font 'Helvetica, 12'")
    @gp 
end

function plot_theta_and_suppression()
    provs = ["Alberta", "Ontario", "Manitoba", "Nova Scotia","Quebec", "British Columbia"]
    main_table = Matrix{Any}(nothing, 0, 5) # 

    # i don't know if making whisker plots make sense here 
    # it shows a 'trend' (unless we put provinces on the y-axis)

    #return chain.supp_flu_mean
    # plot the data in two figures
    # @gp "reset"
    # @gp :- "set multiplot layout 1, 2" :-
    # @gp :- gp_config() :-
    # @gp :- "set style data boxplot" :-
    # @gp :- "set style boxplot nooutliers" :-
    # @gp :- "set boxwidth 0.8 relative" :-
    # @gp :- "set style fill transparent solid 0.35 noborder" :-
    # @gp :- "set bars 2.0" :-    # very large caps
    # @gp :- "set tics textcolor rgb '#444444' font 'Helvetica,12'" :-
    # @gp :- "set pointsize 0.8"             # small point
    
    for (i, prov) in enumerate(provs)
        m1 = read_chain("outputs/chain_$(prov)_m1.jls")
        chain = chain_analysis(m1, prov, plot_figs = false)
        θstr = Printf.@sprintf("%.2f (%.2f–%.2f)", chain.meanθ, chain.lowθ, chain.highθ)
        fstr = Printf.@sprintf("%.0f%% (%.0f–%.0f%%)", chain.supp_flu_mean * 100, chain.supp_flu_low * 100, chain.supp_flu_high * 100)
        cstr = Printf.@sprintf("%.0f%% (%.0f–%.0f%%)", chain.supp_cov_mean * 100, chain.supp_cov_low * 100, chain.supp_cov_high * 100)
        rstr = Printf.@sprintf("%.0f%% (%.0f–%.0f%%)", chain.supp_rsv_mean * 100, chain.supp_rsv_low * 100, chain.supp_rsv_high * 100)
        row = [prov θstr fstr cstr rstr]
        main_table = vcat(main_table, row)
        # @gp :- 1 [i*2] [chain.supp_flu_mean] [chain.supp_flu_low] [chain.supp_flu_high] "with errorbars notitle ls 1 lw 2 pt 7" :-
        # @gp :- 1 [i*2-0.35] [chain.supp_cov_mean] [chain.supp_cov_low] [chain.supp_cov_high] "with errorbars notitle ls 2 lw 2 pt 7" :-
        # @gp :- 1 [i*2+0.35] [chain.supp_rsv_mean] [chain.supp_rsv_low] [chain.supp_rsv_high] "with errorbars notitle ls 3 lw 2 pt 7" :-
    end

    return pretty_table(main_table, backend=:markdown, 
        column_labels = ["Province", "Theta (95% CI)", "Flu", "Covid", "RSV"],
        alignment = [:l, :l, :l, :l, :l])
end

function plot_shedding_kernels() 
    @gp "reset" 
    @gp :- gp_config() :-
    @gp :- "set xlabel 'Days since infection'" :-
    @gp :- "set ylabel 'Relative Shedding'" :-
    #@gp :- "set key spacing 1.2" :-
    #@gp :- "set key width 2" :-
    #@gp :- "set key height 1" :-
    @gp :- "set xtics 1" :- 
    @gp :- "set key noopaque" :- 
    @gp :- "set key Left reverse" :- 
    @gp :- 1:length(K_FLU) K_FLU "smooth csplines with lines title 'Influenza' ls 1" :-
    @gp :- 1:length(K_COV) K_COV "smooth csplines with lines title 'SARS-CoV-2' ls 2" :-
    @gp :- 1:length(K_RSV) K_RSV "smooth csplines with lines title 'RSV' ls 3" :-
    Gnuplot.save("outputs/shedding_kernels.svg", term="svg enhanced size 1000, 400 font 'Helvetica, 16'")
    Gnuplot.save("outputs/shedding_kernels.pdf", term="pdfcairo enhanced size 5in,2in font 'Helvetica, 12'")
    println("plot set, call using @gp in terminal to display.")
    @gp
end

function plot_incidence_curves() 
    provs = ["Alberta", "Ontario", "Manitoba", "Nova Scotia","Quebec", "British Columbia"]
    @gp "reset"
    @gp :- gp_config() :-
    @gp :- "set multiplot layout 6, 3" :-
    @gp :- "set label 101 'Alberta' at screen 0.075, 1.00-0.015 left font 'Helvetica,12' tc rgb '#444444'" :-
    @gp :- "set label 102 'Ontario' at screen 0.075, 0.8333-0.015 left font 'Helvetica,12' tc rgb '#444444'" :-
    @gp :- "set label 103 'Manitoba' at screen 0.075, 0.6667-0.015 left font 'Helvetica,12' tc rgb '#444444'" :-
    @gp :- "set label 104 'Nova Scotia' at screen 0.075, 0.5000-0.015 left font 'Helvetica,12' tc rgb '#444444'" :-
    @gp :- "set label 105 'Quebec' at screen 0.075, 0.3333-0.015 left font 'Helvetica,12' tc rgb '#444444'" :-
    @gp :- "set label 106 'British Columbia' at screen 0.075, 0.1667-0.015 left font 'Helvetica,12' tc rgb '#444444'" :-
    @gp :- "set tmargin 2" :-
    @gp :- "set style fill transparent solid 0.35 noborder" :-

    @gp :- "unset xtics" :-
    # for i = 1:52
    #     @gp :- "set xtics add ($i $(7*i))" :-
    # end
    #@gp :- "set xtics ('Week 1' 52)" :-
    #@gp :- "set xtics 14" :-
    #@gp :- "set xtics format '%g'" :-

    tf = ((8*7)+1):421  # gives me 365 days, 8 weeks of wind-in
    for (prov, i) in zip(provs, 1:3:18)
        _m0 = read_chain("outputs/chain_$(prov)_null.jls")
        _m1 = read_chain("outputs/chain_$(prov)_m1.jls")
        m0 = chain_analysis(_m0, prov, plot_figs = false)
        m1 = chain_analysis(_m1, prov, plot_figs = false)

        mean_inc_flu_m0 = vec(mean(m0.inc_flu[:, tf], dims = 1))
        lo_inc_flu_m0 = [quantile(col, 0.05) for col in eachcol(m0.inc_flu[:, tf])]
        hi_inc_flu_m0 = [quantile(col, 0.975) for col in eachcol(m0.inc_flu[:, tf])]
        mean_inc_flu = vec(mean(m1.inc_flu[:, tf], dims = 1))
        lo_inc_flu = [quantile(col, 0.05) for col in eachcol(m1.inc_flu[:, tf])]
        hi_inc_flu = [quantile(col, 0.975) for col in eachcol(m1.inc_flu[:, tf])]

        mean_inc_cov_m0 = vec(mean(m0.inc_cov[:, tf], dims = 1))
        lo_inc_cov_m0 = [quantile(col, 0.05) for col in eachcol(m0.inc_cov[:, tf])]
        hi_inc_cov_m0 = [quantile(col, 0.975) for col in eachcol(m0.inc_cov[:, tf])]
        mean_inc_cov = vec(mean(m1.inc_cov[:, tf], dims = 1))
        lo_inc_cov = [quantile(col, 0.05) for col in eachcol(m1.inc_cov[:, tf])]
        hi_inc_cov = [quantile(col, 0.975) for col in eachcol(m1.inc_cov[:, tf])]

        mean_inc_rsv_m0 = vec(mean(m0.inc_rsv[:, tf], dims = 1))
        lo_inc_rsv_m0 = [quantile(col, 0.05) for col in eachcol(m0.inc_rsv[:, tf])]
        hi_inc_rsv_m0 = [quantile(col, 0.975) for col in eachcol(m0.inc_rsv[:, tf])]
        mean_inc_rsv = vec(mean(m1.inc_rsv[:, tf], dims = 1))
        lo_inc_rsv = [quantile(col, 0.05) for col in eachcol(m1.inc_rsv[:, tf])]
        hi_inc_rsv = [quantile(col, 0.975) for col in eachcol(m1.inc_rsv[:, tf])]
        
        xn = length(mean_inc_flu)   
        @gp :- i 1:xn lo_inc_flu hi_inc_flu "with filledcurves notitle ls 1" :-
        @gp :- i 1:xn mean_inc_flu "with lines title 'M1' ls 1" :-
        @gp :- i 1:xn lo_inc_flu_m0 hi_inc_flu_m0 "with filledcurves notitle ls 2" :-
        @gp :- i 1:xn mean_inc_flu_m0 "with lines title 'M0' ls 2" :-
        
        @gp :- (i+1) 1:xn lo_inc_cov hi_inc_cov "with filledcurves notitle ls 1" :-
        @gp :- (i+1) 1:xn mean_inc_cov "with lines title 'M1' ls 1" :-
        @gp :- (i+1) 1:xn lo_inc_cov_m0 hi_inc_cov_m0 "with filledcurves notitle ls 2" :-
        @gp :- (i+1) 1:xn mean_inc_cov_m0 "with lines title 'M0' ls 2" :-

        @gp :- (i+2) 1:xn lo_inc_rsv hi_inc_rsv "with filledcurves notitle ls 1" :-
        @gp :- (i+2) 1:xn mean_inc_rsv "with lines title 'M1' ls 1" :-
        @gp :- (i+2) 1:xn lo_inc_rsv_m0 hi_inc_rsv_m0 "with filledcurves notitle ls 2" :-
        @gp :- (i+2) 1:xn mean_inc_rsv_m0 "with lines title 'M0' ls 2" :-
    end
    Gnuplot.save("outputs/incidence_curves.pdf", term="pdfcairo enhanced size 6in,9in font 'Helvetica, 12'")
    @gp

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