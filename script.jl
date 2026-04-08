using DataFrames, CSV
using TidierData
using Random, StatsBase, Distributions
using DifferentialEquations 
using Turing, LogDensityProblems
using MCMCChains, KernelDensity
using SavitzkyGolay # filter
using Gnuplot, PrettyTables, TerminalPager
using Dates
Gnuplot.options.term = "sixelgd truecolor enhanced size 1000,400 scroll font arial 10"

# when adding new packages, avoid precompilation by
# pkg> add --preserve=all Example 
# and see https://pkgdocs.julialang.org/dev/environments/#Controlling-Auto-precompilation

function make_incidence(X) 
    y = similar(X)
    y .= X - circshift(X, 1)    
    y[1] = y[2]
    return y
end

# helper function to apply the savitzky-golay in the TidierData pipeline.
apply_sg(vec) = savitzky_golay(vec, 7, 1).y

function province_params(name) 
    if name == "Regina"
        N = 285_039
        immunity_flu = 0.45
        immunity_cov = 0.70  
        immunity_rsv = 0.55
    elseif name == "British Columbia"
        N = 5_658_528    
        immunity_flu = 0.45
        immunity_cov = 0.70  
        immunity_rsv = 0.55
    elseif name == "Ontario"
        N = 15_000_000
        immunity_flu = 0.45
        immunity_cov = 0.70  
        immunity_rsv = 0.55
    else
        error("Province not recognized. Please add parameters for $name.")
    end
    return N, immunity_flu, immunity_cov, immunity_rsv
end

function read_province(name) 
    # provinces = [
    #     "Alberta",
    #     "British Columbia",
    #     "Manitoba",
    #     "New Brunswick",
    #     "Newfoundland and Labrador",
    #     "Nova Scotia",
    #     "Ontario",
    #     "Prince Edward Island",
    #     "Quebec",
    #     "Saskatchewan"
    # ]
    # unable to do: New Brunswick and Sask 
    # Newfoundland, PEI have slightly less data. 
    println("getting data for province: $name")

    df = DataFrame(CSV.File("canada_wastewater.csv"))
    # clean up a little 
    df = @chain df begin
        @mutate(weekstart = Date(weekstart, "mm/dd/yyyy"))
        @filter(province == !!name)
        @filter(ismissing(site) && ismissing(city))
        @filter(measureid in ("fluA", "fluB", "covN2", "rsv"))
        # see note above for why epiweek 26
        @filter(EpiYear == 2024 && EpiWeek >= 26 || EpiYear == 2025 && EpiWeek <= 34) 
        @select(pruid, site, city, province, weekstart, EpiYear, EpiWeek, measureid, w_avg, populationcoverage)
        @pivot_wider(df, names_from = measureid, values_from = w_avg)
        @mutate(flu = fluA + fluB) # combine fluA and fluB into one signal.
        @mutate(cov_sg = ~apply_sg(covN2)) # the ~ means not to vectorize. 
        @mutate(flu_sg = ~apply_sg(flu)) # the ~ means not to vectorize. 
        @mutate(rsv_sg = ~apply_sg(rsv)) # the ~ means not to vectorize. 
    end

    # calculate how many obs over LOD 
    n_flu_above = sum(skipmissing(df.flu .> minimum(df.flu)))
    n_rsv_above = sum(skipmissing(df.rsv .> minimum(df.rsv)))
    n_total = nrow(df)
    println("Flu above LOD: $n_flu_above / $n_total weeks ($(round(100*n_flu_above/n_total, digits=1))%)")
    println("RSV above LOD: $n_rsv_above / $n_total weeks ($(round(100*n_rsv_above/n_total, digits=1))%)")
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

    return df
end

function sym_defn_ode(du, u, p, t)
    # definitions of SEIR models for each covid, flu, rsv virus 
    # parameters are p = [βᵢ, βᵥ, βᵣ, θ, N] for now (update as needed).
    # the theta is the cross-immunity factor, and N is the total population. 

    # if changing these parameters, change in the analysis function also
    σᵥ = 1/3.0; γᵥ = 1/5.0  
    σᵢ = 1/2.0; γᵢ = 1/4.0
    σᵣ = 1/4.0; γᵣ = 1/7.0

    # v: covid, i: influenza, r: rsv
    Sᵢ, Eᵢ, Iᵢ, Rᵢ, 
    Sᵥ, Eᵥ, Iᵥ, Rᵥ,
    Sᵣ, Eᵣ, Iᵣ, Rᵣ = u

    βᵢ, βᵥ, βᵣ, θ, N = p # unpack parameters

    # calculate the forcce of infection 
    sᵢ = exp(-θ * ((Iᵥ + Iᵣ) / N))
    λᵢ = βᵢ * (Iᵢ / N) * sᵢ

    sᵥ = exp(-θ * ((Iᵢ + Iᵣ) / N))
    λᵥ = βᵥ * (Iᵥ / N) * sᵥ

    sᵣ = exp(-θ * ((Iᵢ + Iᵥ) / N))
    λᵣ = βᵣ * (Iᵣ / N) * sᵣ
    
    #flu 
    du[1] = -λᵢ * Sᵢ   
    du[2] = λᵢ * Sᵢ - σᵢ * Eᵢ
    du[3] = σᵢ * Eᵢ - γᵢ * Iᵢ
    du[4] = γᵢ * Iᵢ

    #covid
    du[5] = -λᵥ * Sᵥ
    du[6] = λᵥ * Sᵥ - σᵥ * Eᵥ
    du[7] = σᵥ * Eᵥ - γᵥ * Iᵥ
    du[8] = γᵥ * Iᵥ

    #rsv
    du[9] = -λᵣ * Sᵣ
    du[10] = λᵣ * Sᵣ - σᵣ * Eᵣ
    du[11] = σᵣ * Eᵣ - γᵣ * Iᵣ
    du[12] = γᵣ * Iᵣ

    # incidence compartments (cumulative) for each virus (remember to take difference to get daily incidence)
    du[13] = σᵢ * Eᵢ  # influenza incidence
    du[14] = σᵥ * Eᵥ  # covid incidence
    du[15] = σᵣ * Eᵣ  # rsv incidence
    
    return nothing # using in-memory update of du 
end

# run ode model with some fixed parameters to see the dynamics 
function simple_ode_model(θ = 0.0) 
    N = 10000  # test population
    u₀ = [9999, 0, 10, 0,  # influenza
          9999, 0, 10, 0,  # covid
          9999, 0, 10, 0, # rsv
          0, 0, 0]  # incidence compartments
    tspan = (0.0, 365) 
    # these parameters are just for testing, not based on literature.
    # they make all three virus have positive incidence rate 
    # which allows for testing theta
    params = [0.28, 0.22, 0.18, θ, N] 
    #Pass to solvers
    prob = ODEProblem(sym_defn_ode, u₀, tspan, params)
    _sol = solve(prob, Tsit5())
    sol = reduce(hcat, _sol.u)

    #plot 
    println("plot set")
    @gp "reset" 
    @gp :- "set multiplot layout 2, 3" :- # 1 row, 3 columns
    @gp :- 1 _sol.t sol[1, :] "with lines title 'S' lc rgb 'black'" :-
    @gp :- 1 _sol.t sol[2, :] "with lines title 'E' lc rgb 'blue'" :-
    @gp :- 1 _sol.t sol[3, :] "with lines title 'I' lc rgb 'red'" :-
    @gp :- 1 _sol.t sol[4, :] "with lines title 'R' lc rgb 'green'" :-

    @gp :- 2 _sol.t sol[5, :] "with lines title 'S' lc rgb 'black'" :-
    @gp :- 2 _sol.t sol[6, :] "with lines title 'E' lc rgb 'blue'" :-
    @gp :- 2 _sol.t sol[7, :] "with lines title 'I' lc rgb 'red'" :-
    @gp :- 2 _sol.t sol[8, :] "with lines title 'R' lc rgb 'green'" :-

    @gp :- 3 _sol.t sol[9, :] "with lines title 'S' lc rgb 'black'" :-
    @gp :- 3 _sol.t sol[10, :] "with lines title 'E' lc rgb 'blue'" :-
    @gp :- 3 _sol.t sol[11, :] "with lines title 'I' lc rgb 'red'" :-
    @gp :- 3 _sol.t sol[12, :] "with lines title 'R' lc rgb 'green'" :-

    @gp :- 4 _sol.t make_incidence(sol[13, :]) "with lines title 'Incidence (flu)' lc rgb 'orange' lw 2" :-
    @gp :- 4 _sol.t make_incidence(sol[14, :]) "with lines title 'Incidence (covid)' lc rgb 'purple' lw 2" :-
    @gp :- 4 _sol.t make_incidence(sol[15, :]) "with lines title 'Incidence (rsv)' lc rgb 'brown' lw 2" :-
    # @gp  $ call @gp in terminal to display the plot.
    return sol
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
@model function inference_model(
        obs_ww_flu, # now length = wind-in weeks + main_weeks
        obs_ww_cov, 
        obs_ww_rsv,
        lod_ww_flu, # LOD for flu
        lod_ww_rsv, # LOD for RSV, Covid not needed, strong signal
        pop_size,
        immunity_flu, # pre-immunity levels
        immunity_cov,
        immunity_rsv
    )
    # obs_ww_flu, obs_ww_cov, obs_ww_rsv are the observed wastewater signals for each virus,
    # which we compare to the simulated wastewater signal from the ODE.    
    # parameters that will be sampled: theta, beta (3 of them), scaling factors (for WW) + other intermediates if required


    # # regina
    # # log_M_r ~ Normal(-4.3, 0.15)
    # # M_r = exp(log_M_r)
    # #Bc 
    # log_M_r ~ Normal(-4.3, 0.10)
    # M_r = exp(log_M_r)

    # After this change, M is no longer in gc/L per person — it is now in gc/L per fraction of population infected
    # The scale changes completely. Back-calculating from your BC COVID data:
    # M_c = \frac{W_{\text{peak}}}{(I_{\text{peak}}/N)} \approx \frac{50 \text{ gc/L}}{0.002} = 25{,}000 \text{ gc/L per prevalence fraction}
    # \log M_c \approx \log(25000) \approx 10.1
    # So  M priors need to be completely recentred:
    # new — correct scale after per-capita normalisation
    log_M_c ~ Normal(10.0, 0.5)
    log_M_f ~ Normal(9.0, 0.5)   # flu sheds less efficiently per capita
    log_M_r ~ Normal(9.0, 0.5)
    M_c = exp(log_M_c)
    M_f = exp(log_M_f)
    M_r = exp(log_M_r)

    # switching the likelihood to Normal. Initially was NB, but that's for discrete data. 
    # observation noise on log scale, Exponential(0.3) has mean 0.3, 
    # which implies ~30% multiplicative noise, reasonable for WW data (NO, see obsidian notes for why this is too high and leads to poor inference).
    #sigma_obs ~ Exponential(0.5)   # observation noise on log scale
    #  update: even this is too high producing a posterior that suggests multiplicative noise of ~7x 
    # sigma_obs ~ Truncated(Normal(0, 0.5), 0, Inf)

    # better — hard ceiling that reflects smoothed data reality
    # sigma_obs ~ Truncated(Normal(0.8, 0.2), 0.05, Inf) # regina
    
    # BC 
    #sigma_obs ~ Truncated(Normal(0.6, 0.15), 0.1, 1.2) # BC 
    # bc may require sepratate sigma_obs for each virus, because the smoothed data suggests covid is less noisy than flu and rsv (see obsidian notes for details on how this affects inference and model fit).
    sigma_cov ~ truncated(Normal(0.5, 0.15), 0.1, 0.9)   # provincial COVID well-measured
    sigma_flu ~ truncated(Normal(0.8, 0.2),  0.1, 1.3)   # flu noisier
    sigma_rsv ~ truncated(Normal(0.9, 0.2),  0.1, 1.5)   # RSV noisiest — allow higher

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

    # setup the ODE for running 

    # beta values: the posterior of these should be informed 
    # by literature. Since the model is simple, 
    # R = beta / recovery. 

    # βᵢ ~ Truncated(LogNormal(log(0.3), 0.3), 0.05, 1.0)  #mean 0.31 
    # βᵥ ~ Truncated(LogNormal(log(0.35), 0.3), 0.05, 1.0) #mean 0.37
    # βᵣ ~ Truncated(LogNormal(log(0.25), 0.3), 0.05, 1.0) #mean 0.26
    
    # run april 6, okay, large final size
    # βᵢ ~ Truncated(LogNormal(log(0.34), 0.15), 0.15, 0.6)
    # βᵥ ~ Truncated(LogNormal(log(0.87), 0.2), 0.65, 1.5)
    # βᵣ ~ Truncated(LogNormal(log(0.24), 0.15), 0.1, 0.5) #mean 0.26

    βᵢ ~ truncated(LogNormal(log(0.62), 0.15), 0.35, 1.0)
    βᵥ ~ truncated(LogNormal(log(0.87), 0.2), 0.65, 1.5)
    βᵣ ~ truncated(LogNormal(log(0.43), 0.15), 0.25, 0.8) #mean 0.26

    # initial values 
    #i0c ~ truncated(Normal(500.0, 300.0), 1.0, pop_size * 0.05) # reigna
    i0i ~ truncated(Normal(10.0, 20.0), 1.0, 200) # 
    i0c ~ truncated(Normal(3000.0, 1500.0), 10.0, pop_size * 0.05)
    i0r ~ truncated(Normal(5.0, 10.0), 1.0, 200) 
    
    # susceptible 
    s0i = (1.0 - immunity_flu) * pop_size
    s0c = (1.0 - immunity_cov) * pop_size
    s0r = (1.0 - immunity_rsv) * pop_size
    
    # recovered compartments
    r0i = immunity_flu * pop_size
    r0c = immunity_cov * pop_size
    r0r = immunity_rsv * pop_size

    initial_cond = [
        s0i - i0i, 0.0, i0i, r0i,   # flu:   S E I R
        s0c - i0c, 0.0, i0c, r0c,   # covid: S E I R
        s0r - i0r, 0.0, i0r, r0r,   # rsv:   S E I R
        0.0, 0.0, 0.0               # incidence
    ]
    
    weeks = length(obs_ww_flu) # or obs_ww_cov, obs_ww_rsv, they should all be the same length.
    days = weeks * 7.0  
    tspan = (0.0, days)

    θ ~ Truncated(Normal(0, 2.0), 0, Inf) 
    params = [βᵢ, βᵥ, βᵣ, θ, pop_size] # params vector: [βᵢ, βᵥ, βᵣ, θ, N]
   
    # solve the ODE 
    prob = ODEProblem(sym_defn_ode, initial_cond, tspan, params)
    _sol = solve(prob, Tsit5(), saveat=1.0)
    sol = reduce(hcat, _sol.u)

    #_sol = solve(prob, Rodas5P(), saveat=1.0, maxiters=1e6)
    # if _sol.retcode != ReturnCode.Success
    #     Turing.@addlogprob! -Inf
    #     return
    # end

    # get the ODE simulated incidence (this will be in days)
    Iᵢ_sim = make_incidence(sol[13, :])
    Iᵥ_sim = make_incidence(sol[14, :])
    Iᵣ_sim = make_incidence(sol[15, :])

    Wᵢ_sim = sym_convolve_seir(Iᵢ_sim ./ pop_size, K_FLU) * M_f
    Wᵥ_sim = sym_convolve_seir(Iᵥ_sim ./ pop_size, K_COV) * M_c
    Wᵣ_sim = sym_convolve_seir(Iᵣ_sim ./ pop_size, K_RSV) * M_r
    
    # number of periods to windin (meaning, do not compute likelyhood of flu/rsv WW for these weeks since WW signal is low and noisy, and we don't want to fit to noise in this period).
    N_WINDIN = 9

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
            obs_ww_cov[w] ~ LogNormal(log(mu_c), sigma_cov)
        end

        # for flu and rsv, ignore the wind-in period 
        # this is because WW obs during this time is low 
        if w > N_WINDIN
            if !ismissing(obs_ww_flu[w])
                if obs_ww_flu[w] <= lod_ww_flu   # at LOD: P(X <= LOD)
                    #Turing.@addlogprob! logcdf(LogNormal(log(mu_i), sigma_obs), lod_flu)
                    Turing.@addlogprob! logpdf(LogNormal(log(mu_i), sigma_flu * 2.0), lod_ww_flu)
                else
                    obs_ww_flu[w] ~ LogNormal(log(mu_i), sigma_flu)
                end
            end
            if !ismissing(obs_ww_rsv[w]) 
                if obs_ww_rsv[w] <= lod_ww_rsv   # at LOD: P(X <= LOD)
                    #Turing.@addlogprob! logcdf(LogNormal(log(mu_r),  sigma_obs), lod_rsv)
                    Turing.@addlogprob! logpdf(LogNormal(log(mu_r), sigma_rsv * 2.0),  lod_ww_rsv)
                else
                    obs_ww_rsv[w] ~ LogNormal(log(mu_r),  sigma_rsv) 
                end
            end
        end
    end
end

function run_inference() 
    # get WW data 
    province = "Ontario"
    df = read_province(province) 
    display(@gp)
    obs_flu = df.flu_sg
    obs_covid = df.cov_sg
    obs_rsv = df.rsv_sg
    @assert length(df.flu_sg) == length(df.cov_sg) == length(df.rsv_sg)
    
    lod_flu = minimum(obs_flu)
    lod_rsv = minimum(obs_rsv)

    # initial conditions and population size for the ODE model
    N, immunity_flu, immunity_cov, immunity_rsv = province_params(province)
    N = round(Int64, N * unique(df.populationcoverage)[1])
    println("population of $province, adjusted: $N")

    # step size that seems to work well: ϵ = 0.00625
    _m1 = inference_model(obs_flu, obs_covid, obs_rsv, lod_flu, lod_rsv, N, immunity_flu, immunity_cov, immunity_rsv)
    #return Turing.DynamicPPL.DebugUtils.model_warntype(_m1)    
    # init_ϵ = 0.00625 works well
    m1 = sample(_m1, NUTS(), MCMCThreads(), 1000, 4)  # 1000 samples, 4 chains for parallel sampling
    println("m1 model finished (with non-zero theta), summary: ")
    display(summarystats(m1))
    
    # _m0 = inference_model(obs_flu, obs_covid, obs_rsv, N, Iᵢ, Iᵥ, Iᵣ, true, true)
    # m0 = sample(_m0, NUTS(init_ϵ = 0.00625), MCMCThreads(), 5000, 4)  # 1000 samples, 4 chains for parallel sampling
    # println("m0 model finished (with zero theta), summary: ")
    # display(summarystats(m0))

    # return _m1, m1, _m0, m0
    return _m1, m1
end 

function plot_traceplots(chain, param) 
    #param_names = chain.name_map.parameters
    #println(param_names)
    #x = chain[:, Symbol(param), :]
    #numchains = size(x, 2)
 
end

function plot_posterior_distribution(chain) 
    param_names = chain.name_map.parameters
    # [:βᵢ, :βᵥ, :β]
    println(param_names)
    x = vec(chain[:, :βᵣ, :])   # flatten all chains
    kd = kde(x)
    @gp "reset" 
    @gp :- kd.x kd.density "w l lw 2 lc rgb 'blue'"
    #@gp :- 1 
    #@gp
end


function plot_predictive(chain)
    # performs the analysis of the posterior samples
    # and plots the predictive distribution of the wastewater signal,
    # as well as the inferred incidence and prevalence curves.
    
    # get the raw data 
    df = read_province("Ontario")
    obs_flu = df.flu_sg
    obs_covid = df.cov_sg
    obs_rsv = df.rsv_sg
    weeks = length(obs_flu)
    days = weeks * 7 + 1

    N, immunity_flu, immunity_cov, immunity_rsv = province_params("Ontario") # initial conditions for the ODE, based on province parameters.
    N = round(Int64, N * unique(df.populationcoverage)[1]) # adjust the effective population
    println("population of Ontario, adjusted: $N")

    #print the stats of the Bayesian chains
    display(summarystats(chain))    
    
    # calculate the beta <-> R0 relationship at the posterior mean 
    beta_i = mean(m1[:, :βᵢ, :])
    beta_v = mean(m1[:, :βᵥ, :])
    beta_r = mean(m1[:, :βᵣ, :])
    γᵢ = 1/4.0 # from the ODE defintions
    γᵥ = 1/5.0  
    γᵣ = 1/7.0
    println("R0, flu: $(round(beta_i / γᵢ; digits=2)), eff R: $(round(beta_i / γᵢ * (1 - immunity_flu); digits=2)), R range: 1.2-1.4")
    println("R0, COVID: $(round(beta_v / γᵥ; digits=2)), eff R: $(round(beta_v / γᵥ * (1 - immunity_cov); digits=2)), R range: 2.0-3.0")
    println("R0, RSV: $(round(beta_r / γᵣ; digits=2)), eff R: $(round(beta_r / γᵣ * (1 - immunity_rsv); digits=2)), R range: 1.5-2.5")

    # plot the predictive distribution of the wastewater signal 
    # based on the posterior samples of the parameters.
    nsamples = 200 
    samples = get(sample(chain, nsamples); section=:parameters)

    # empty matrices to store WW results
    ww_cov_mat = zeros(nsamples, weeks)
    ww_flu_mat = zeros(nsamples, weeks)
    ww_rsv_mat = zeros(nsamples, weeks)

    # empty matrices to store incidence reuslts (in days)
    inc_cov_mat = zeros(nsamples, weeks * 7 + 1) 
    inc_flu_mat = zeros(nsamples, weeks * 7 + 1)
    inc_rsv_mat = zeros(nsamples, weeks * 7 + 1)

    # empty matrices to store prevalnece reuslts (in days)
    prev_cov_mat = zeros(nsamples, weeks * 7 + 1) 
    prev_flu_mat = zeros(nsamples, weeks * 7 + 1)
    prev_rsv_mat = zeros(nsamples, weeks * 7 + 1)

    for i in 1:nsamples
        # extract the parameters for this sample
        θ = samples[:θ][i]
        βᵢ = samples[:βᵢ][i]
        βᵥ = samples[:βᵥ][i]
        βᵣ = samples[:βᵣ][i]
        M_c = exp(samples[:log_M_c][i])
        M_f = exp(samples[:log_M_f][i])
        M_r = exp(samples[:log_M_r][i])
        i0c = samples[:i0c][i] 
        i0r = samples[:i0r][i]
        i0i = samples[:i0i][i]
        rsv_shape = samples[:rsv_shape][i]
        rsv_scale = samples[:rsv_scale][i]
        flu_shape  = samples[:flu_shape][i]
        flu_scale  = samples[:flu_scale][i]


        # susceptible 
        s0i = (1.0 - immunity_flu) * N
        s0c = (1.0 - immunity_cov) * N
        s0r = (1.0 - immunity_rsv) * N
        
        # recovered compartments
        r0i = immunity_flu * N
        r0c = immunity_cov * N
        r0r = immunity_rsv * N

        u₀ = [
            s0i - i0i, 0.0, i0i, r0i,   # flu:   S E I R
            s0c - i0c, 0.0, i0c, r0c,   # covid: S E I R
            s0r - i0r, 0.0, i0r, r0r,   # rsv:   S E I R
            0.0, 0.0, 0.0               # incidence
        ]

        params = [βᵢ, βᵥ, βᵣ, θ, N]
        tspan  = (0.0, Float64(weeks * 7))
        prob   = ODEProblem(sym_defn_ode, u₀, tspan, params)
        sol    = solve(prob, Tsit5(), saveat=1.0)
        sol_m  = reduce(hcat, sol.u)

        # store prevalence results for this sample
        prev_flu_mat[i, :] = sol_m[3, :]
        prev_cov_mat[i, :] = sol_m[7, :]
        prev_rsv_mat[i, :] = sol_m[11, :]
        
        #store incidence
        Iᵢ = make_incidence(sol_m[13, :])
        Iᵥ = make_incidence(sol_m[14, :])
        Iᵣ = make_incidence(sol_m[15, :])
        inc_flu_mat[i, :] = Iᵢ
        inc_cov_mat[i, :] = Iᵥ
        inc_rsv_mat[i, :] = Iᵣ
        #println(Iᵥ[1])

        K_RSV_s = make_kernel(rsv_shape, rsv_scale, 21)
        K_FLU_s = make_kernel(flu_shape, flu_scale, 21)

        Wf = sym_convolve_seir(Iᵢ ./ N, K_FLU_s) * M_f
        Wc = sym_convolve_seir(Iᵥ ./ N, K_COV) * M_c
        Wr = sym_convolve_seir(Iᵣ ./ N, K_RSV_s) * M_r

        for w in 1:weeks
            idx_end = w * 7
            idx_end > size(Wc, 1) && continue
            ww_cov_mat[i, w] = mean(Wc[idx_end-6:idx_end])
            ww_flu_mat[i, w] = mean(Wf[idx_end-6:idx_end])
            ww_rsv_mat[i, w] = mean(Wr[idx_end-6:idx_end])
        end
    end

    # temporal suppression calculation
    _cov = vec(mean(prev_cov_mat, dims = 1))
    _flu = vec(mean(prev_flu_mat, dims = 1))
    _rsv = vec(mean(prev_rsv_mat, dims = 1))
    θ_mean = mean(samples[:θ])

    supp_cov = [1 - exp(-θ_mean * (_flu[t] + _rsv[t])/N) for t in 1:days]
    supp_rsv = [1 - exp(-θ_mean * (_flu[t] + _cov[t])/N) for t in 1:days]
    supp_flu = [1 - exp(-θ_mean * (_cov[t] + _rsv[t])/N) for t in 1:days]

    # suppression_cov = 1 - exp(-mean(samples[:θ]) * (peak_flu + peak_rsv)/N)
    # suppression_rsv = 1 - exp(-mean(samples[:θ]) * (peak_flu + peak_cov)/N)
    # suppression_flu = 1 - exp(-mean(samples[:θ]) * (peak_cov + peak_rsv)/N)
    # println("Estimated COVID suppression due to cross-immunity: $(round(mean(suppression_cov) * 100, digits=1))%")
    # println("Estimated RSV suppression due to cross-immunity: $(round(mean(suppression_rsv) * 100, digits=1))%")
    # println("Estimated Flu suppression due to cross-immunity: $(round(mean(suppression_flu) * 100, digits=1))%")
    
    #final size:
    println("final size flu: $(sum(vec(mean(inc_flu_mat, dims = 1))) )")
    println("final size covid: $(sum(vec(mean(inc_cov_mat, dims = 1))) )")
    println("final size rsv: $(sum(vec(mean(inc_rsv_mat, dims = 1))) )")
    
    
    #plot the posteriors and trace plots 
    @gp "reset"
    @gp :- "set multiplot layout 3, 1" :-
    @gp :- 1 1:weeks obs_flu "with lines title 'Obs Flu' lw 3 lc rgb 'black'" :-
    @gp :- 2 1:weeks obs_covid "with lines title 'Obs Covid' lw 3 lc rgb 'black'" :-
    @gp :- 3 1:weeks obs_rsv "with lines title 'Obs RSV' lw 3 lc rgb 'black'" :-
    @gp :- 1 1:weeks vec(mean(ww_flu_mat, dims = 1)) "with lines title 'Pred Flu' lw 2 lc rgb 'blue'" :-
    @gp :- 2 1:weeks vec(mean(ww_cov_mat, dims = 1)) "with lines title 'Pred Covid' lw 2 lc rgb 'blue'" :-
    @gp :- 3 1:weeks vec(mean(ww_rsv_mat, dims = 1)) "with lines title 'Pred RSV' lw 2 lc rgb 'blue'" :-
    Gnuplot.save("predictive_wastewater.svg", term="svg enhanced size 800,800 font ', 22' background rgb '#FAFAFA'")
    display(@gp)
    
    @gp "reset"
    @gp :- "set multiplot layout 3, 1" :-
    @gp :- 1 1:size(inc_flu_mat, 2) cumsum(vec(mean(inc_flu_mat, dims = 1))) "with lines title 'Obs Flu' lw 3 lc rgb 'black'" :-
    @gp :- 2 1:size(inc_cov_mat, 2) cumsum(vec(mean(inc_cov_mat, dims = 1))) "with lines title 'Obs Covid' lw 3 lc rgb 'black'" :-
    @gp :- 3 1:size(inc_rsv_mat, 2) cumsum(vec(mean(inc_rsv_mat, dims = 1))) "with lines title 'Obs RSV' lw 3 lc rgb 'black'" :-
    Gnuplot.save("final_size.svg", term="svg enhanced size 800,800 font ', 22' background rgb '#FAFAFA'")
    display(@gp)

    @gp "reset" 
    @gp :- 1:days supp_flu "with lines title 'Flu Suppression' lw 2 lc rgb 'red'" :-
    @gp :- 1:days supp_cov "with lines title 'Covid Suppression' lw 2 lc rgb 'blue'" :-
    @gp :- 1:days supp_rsv "with lines title 'RSV Suppression' lw 2 lc rgb 'green'" :-
    Gnuplot.save("supressive_temporal.svg", term="svg enhanced size 800,800 font ', 22' background rgb '#FAFAFA'")
    display(@gp)

    @gp "reset"
    #@gp :- "set multiplot layout $numchains, 1" :-
    @gp :-  chain[:, :θ, 1] "with lines title 'Chain 1' lc rgb 'blue'" :-
    @gp :-  chain[:, :θ, 2] "with lines title 'Chain 2' lc rgb 'red'" :-
    @gp :-  chain[:, :θ, 3] "with lines title 'Chain 3' lc rgb 'green'" :-
    @gp :-  chain[:, :θ, 4] "with lines title 'Chain 4' lc rgb 'orange'" :-
    @gp 

    return nothing
end


# run a test version of the model 
# here we generate our own simulated WW from the ODE first 
# and see if we can recover some parameters from the Turing model 
# we use 10000 individuals - this is just to test if the model runs 
# and can recover parameters close to the true value.
# March 2nd test: works flawlessly, can recover parameters close to the true value.
function basic_inference_model(test_theta = 0.0) 
    # this function just runs the Turing model 
    # but only fits up to observed incidence, not wastewater 
    # it's just a test to see if the model even runs. 

    println("theta value: $test_theta")
    # generate the simulated data from ODE 
    # match the parameters in  sym_ode_model(), 
    N = 10000  # test population
    
    # initial 
    u₀ = [9999, 0, 10, 0,  # influenza
          9999, 0, 10, 0,  # covid
          9999, 0, 10, 0, # rsv
          0, 0, 0]  # incidence compartments
    tspan = (1, 364) 
    # test parameters
    tβᵢ = 0.28
    tβᵥ = 0.22
    tβᵣ = 0.18
    tθ = test_theta  # for testing, generate data with no or some cross-immunity, and see if we can recover that.
    params = [tβᵢ, tβᵥ, tβᵣ, tθ, N] 

    #Pass to solvers
    prob = ODEProblem(sym_defn_ode, u₀, tspan, params)
    sol = solve(prob, Tsit5(), saveat=1.0)

    # get the incidence 
    Iᵢ_sim = make_incidence(sol[13, :])
    Iᵥ_sim = make_incidence(sol[14, :])
    Iᵣ_sim = make_incidence(sol[15, :])

    # scaling factors 
    M_f = 2.1
    M_c = 1.8
    M_r = 1.3

    # convert to wastewater signal using convolution with the kernels 
    # Do not convert to weekly data right now - 
    # -> it kills the biological resolution of the shedding kernel.
    #  For example, SARS-CoV-2 shedding peaks around day 4 or 5 after infection;
    #  if you aggregate into weeks first, the math cannot accurately apply that daily shedding delay.
    _Wᵢ_sim = sym_convolve_seir(Iᵢ_sim, K_FLU)
    _Wᵥ_sim = sym_convolve_seir(Iᵥ_sim, K_COV)
    _Wᵣ_sim = sym_convolve_seir(Iᵣ_sim, K_RSV)

    # convert to a weekly signal by taking the mean of each 7 days, 
    # and add some noise to simulate observed data. 
    Wᵢ_sim = mean.(eachcol(reshape(_Wᵢ_sim, 7, :))) * M_f
    Wᵥ_sim = mean.(eachcol(reshape(_Wᵥ_sim, 7, :))) * M_c
    Wᵣ_sim = mean.(eachcol(reshape(_Wᵣ_sim, 7, :))) * M_r

    println("lengths of vectors: 
        length of ODE incidence: $(length(Iᵢ_sim))
        length of ODE simulated WW: $(length(Wᵢ_sim))
        length of weekly converted 'observed' WW: $(length(Wᵢ_sim))
    ")

    # add noise to the WW signal 
    sigma_test = 0.3   # pick a plausible observation noise
    obs_flu  = rand.(LogNormal.(log.(Wᵢ_sim .+ 1e-6), sigma_test))
    obs_covid = rand.(LogNormal.(log.(Wᵥ_sim .+ 1e-6), sigma_test))
    obs_rsv  = rand.(LogNormal.(log.(Wᵣ_sim .+ 1e-6), sigma_test))

    # obs_flu = Float64.(rand.(Poisson.(Wᵢ_sim)))
    # obs_covid = Float64.(rand.(Poisson.(Wᵥ_sim)))
    # obs_rsv =  Float64.(rand.(Poisson.(Wᵣ_sim)))
    
    #return obs_rsv
    ## plot these with incidence curves to see if they look reasonable 
    
    @gp "reset"
    #@gp :- "set multiplot layout 2, 3" :- 
    @gp :- sol.t Iᵢ_sim "with lines title 'Incidence (flu)' lc rgb 'orange' lt 2 dt 2 lw 2" :-
    @gp :- sol.t[1:7:end] obs_flu "with lines title 'Obs WW (flu)' lc rgb 'orange' lt 2 dt 2 lw 2" :-
    @gp :- sol.t Iᵥ_sim "with lines title 'Incidence (covid)' lc rgb 'purple' lw 2" :-
    @gp :- sol.t[1:7:end] obs_covid "with lines title 'Obs WW (covid)' lc rgb 'purple' lt 2 dt 2 lw 2" :-
    @gp :- sol.t Iᵣ_sim "with lines title 'Incidence (rsv)' lc rgb 'blue' lw 2" :-
    @gp :- sol.t[1:7:end] obs_rsv "with lines title 'Obs WW (rsv)' lc rgb 'blue' lt 2 dt 2 lw 2" :-
    println("plot set, call using @gp in terminal to display.") 
    
    # run 
    my_model = inference_model(obs_flu, obs_covid, obs_rsv, N, 10, 10, 10, false)
    println("Starting MCMC Sampling...")
    chain = sample(my_model, NUTS(), 1000)
    return chain
end

function plot_chain(chain)
    theta_chains = chain[:, :θ, :] 
    @gp "reset"
    @gp :- theta_chains[:, 1] "with lines title 'Chain 1' lc rgb 'red'" :-
    @gp :- theta_chains[:, 2] "with lines title 'Chain 2' lc rgb 'blue'" :-
    @gp :- theta_chains[:, 3] "with lines title 'Chain 3' lc rgb 'green'" :-
    @gp :- theta_chains[:, 4] "with lines title 'Chain 4' lc rgb 'orange'" :-
    @gp
end
########################################################################## 
####### TEST MODEL  (basic SIR model with Bayesian parameter inference)
########################################################################## 

# the following are to just test a simple SIR model
# used to test the proof of concept of runing a Bayesian ODE model.
function tst_defn_ode_sys(du, u, p, t) 
    S, I = u     # Unpack state variables
    beta = p[1]  # Unpack our single parameter
    du[1] = -beta * S * I  # dS/dt: Susceptible people getting infected
    du[2] =  beta * S * I  # dI/dt: New infections added to the pool
end

function tst_run_ode_sys(beta) 
    # just run the basic ODE model
    u0 = [99.0, 1.0] 
    tspan = (0.0, 100.0) 
    p = [beta]
    prob = ODEProblem(tst_defn_ode_sys, u0, tspan, p)  # Define the ODE problem
    _sol = solve(prob, Tsit5(), saveat=1.0) # The sampler guesses a beta, runs the ODE, and saves the daily results
    sol = reduce(hcat, _sol.u)  # Convert solution to a matrix for plotting
    I_sim = sol[2, :]
    @gp "reset" 
    @gp :- _sol.t sol[2, :] "with lines notitle lc rgb 'blue'" :-
    println("plot set")
    return _sol
end

# turing model 
@model function tst_defn_turing_model(data, u0, tspan)
    # Bayesian inference 
    # Calling sample(syn_turing, NUTS(), 1000) does the following
    # The algorithm looks at the line beta ~ LogNormal(log(0.1), 1.0). 
    # It picks a random number from this distribution—let's say it guesses $\beta = 0.08$.
    # It plugs $\beta = 0.08$ into the ODEProblem and runs it. 
    # Because 0.08 is higher than the true value of 0.05,
    # the simulation will produce a curve where everyone gets infected way too fast. It might predict that on Day 5, there are 80 infected people.
    # The likelihood: It reaches the for i in 1:length(data) loop. It looks at your obs_data for Day 5. Let's say your data shows only 15 people were infected on Day 5.
    # The algorithm calculates: "What is the probability of observing 15 from a Poisson distribution that expects 80?"
    # That probability is practically zero. The Turing engine assigns a terrible "Likelihood Score" to $\beta = 0.08$.

    # NUTS algorithm: Because the score was terrible, the algorithm uses 
    # calculus (Automatic Differentiation) to figure out which direction 
    # to adjust $\beta$. It realizes it needs to lower the number. 
    # Next time, it might guess $\beta = 0.06$. It gets a better score. 
    # Then it tries $\beta = 0.05$, gets an excellent score, and starts 
    # taking lots of samples around that area.

    # When the script finishes, it will print a table of statistics. Look for the mean of beta in the output.
    # Because true_beta was exactly 0.05, the MCMC sampler's mean should
    # settle extremely close to ~0.049 to 0.051. The std (standard deviation) will tell you how confident the model is in that guess.

    # sometimes the test model fails (since beta is LogNormal)
    # ```
    # sampled dual b: 0.06919122724261408
    # sampled dual b: 0.0040338384347376665
    # ┌ Warning: Verbosity toggle: max_iters
    # │  Interrupted. Larger maxiters is needed. If you are using an integrator for non-stiff ODEs or an automatic switching algorithm (the default), you may want to consider using a method for stiff equations. See the solver pages for more details (e.g. https://docs.sciml.ai/DiffEqDocs/stable/solvers/ode_solve/#Stiff-Problems).
    # └ @ SciMLBase ~/.julia/packages/SciMLBase/obTYa/src/integrator_interface.jl:635
    # sampled dual b: 1338.1957620169487
    # ```

    # specify the prior for beta
    #beta ~ Truncated(LogNormal(log(0.1), 1.0), 0.0, 1.0)
    beta ~ Truncated(Gamma(2, 2), 0.0, 1.0)

    p = [beta]
    prob = ODEProblem(tst_defn_ode_sys, u0, tspan, p)  # Define the ODE problem
    _sol = solve(prob, Tsit5(), saveat=1.0) # The sampler guesses a beta, runs the ODE, and saves the daily results
    sol = reduce(hcat, _sol.u)  # Convert solution to a matrix for plotting
    I_sim = sol[2, :]
   
    # just a print statement since it was failing 
    # if (beta isa Turing.ForwardDiff.Dual)
    #     println("sampled dual b: $(beta.value)")
    # end
    
    # this doesn't work because beta is a Dual number, which is a special type used for automatic differentiation.
    # println("sampled b: $beta")
    # println("typeof: $(beta isa Turing.ForwardDiff.Dual)")
    # #push!(betas, beta)
    
    # Extract the simulated Infected curve based on the current guess for beta
    # We compare the simulated curve to our noisy observed data
    for i in eachindex(data)
        # We add 1e-6 to prevent mathematically impossible Poisson(0) errors
        data[i] ~ Poisson(I_sim[i] + 1e-6) 
    end
end

function tst_run_bayesian_model() 
    # runs the ODE model directly with fixed parameters 
    # then adds noise to the simulated data and 
    # and tries to recover the parameters using Turing.
    u0 = [99.0, 1.0] 
    tspan = (0.0, 100.0) # Run for 20 days
    p = [0.001]  # beta rate parameter
    prob = ODEProblem(tst_defn_ode_sys, u0, tspan, p)  # Define the ODE problem
    _sol = solve(prob, Tsit5(), saveat=1.0)  # Solve the ODE using the Ts
    sol = reduce(hcat, _sol.u)  # Convert solution to a matrix for plotting
    @gp "reset"
    @gp :- _sol.t sol[1, :] "with lines notitle lc rgb 'blue'" :-
    @gp :- _sol.t sol[2, :] "with lines notitle lc rgb 'red'" :- 
    
    # add noise to the data to see if we can recover it / plot it
    true_data = sol[2, :] # 
    #obs_data = rand.(Poisson.(true_data))
    obs_data = round.(Int64, rand.(Exponential.(true_data)))
    #obs_data = rand.(NegativeBinomial.(true_data, 0.4))
    @gp :- _sol.t obs_data "with lines  notitle lc rgb 'black'" :-
    @gp
    #return obs_data
    #return 
    # can we recover beta using Turing?
    my_model = tst_defn_turing_model(obs_data, u0, tspan)
    println("Starting MCMC Sampling...")
    chain = sample(my_model, NUTS(), 1000)
    display(describe(chain))
    return chain
end
    
########################################################################## 
#### A basic SEIR MODEL I used to test the idea of calculating incidence from the ODE 
#### basically needs another compartment in the ODE defn
########################################################################## 

function inc_defn_ode(du, u, p, t)
    # definitions of SEIR models for each covid, flu, rsv virus 
    # parameters are p = [βᵢ, βᵥ, βᵣ, θ, N] for now (update as needed).
    # the theta is the cross-immunity factor, and N is the total population. 

    σᵣ = 1/4.0
    γᵣ = 1/7.0
    Sᵣ, Eᵣ, Iᵣ, Rᵣ = u

    # unpack the parameters, fourth one is theta. 
    βᵣ, N = p
    λᵣ = βᵣ * (Iᵣ / N)
    du[1] = -λᵣ * Sᵣ
    du[2] = λᵣ * Sᵣ - σᵣ * Eᵣ
    du[3] = σᵣ * Eᵣ - γᵣ * Iᵣ
    du[4] = γᵣ * Iᵣ
    du[5] = σᵣ * Eᵣ  # compartment to just hold the incidence (cumulative), so you'll have to take difference
    return nothing
end

function inc_run_ode() 
    u₀ = [9999, 0, 1, 0, 0] # rsv
    tspan = (0.0, 100)
    params = [0.5, 10000]
    #Pass to solvers
    prob = ODEProblem(inc_defn_ode, u₀, tspan, params)
    _sol = solve(prob, Tsit5())
    sol = reduce(hcat, _sol.u)

    gem_incidence = sol[2, :] * 1/4.0 # suggested from AI, but not correctֱ
    com_incidence = sol[5, :] # proper incidence calculated in the ODE (but it's cumulative, so we need to take the difference)
    incidence_2 = make_incidence(com_incidence) # 
    #plot 
    println("plot set")
    @gp "reset" 
    @gp :- "set multiplot layout 1, 1" :- # 1 row, 3 columns
    @gp :- 1 _sol.t sol[1, :] "with lines title 'S' lc rgb 'black'" :-
    @gp :- 1 _sol.t sol[2, :] "with lines title 'E' lc rgb 'blue'" :-
    @gp :- 1 _sol.t sol[3, :] "with lines title 'I' lc rgb 'red'" :-
    @gp :- 1 _sol.t sol[4, :] "with lines title 'R' lc rgb 'green'" :-

    @gp :- 1 _sol.t gem_incidence "with lines title 'Incidence (gem)' lc rgb 'orange' lw 2" :-
    @gp :- 1 _sol.t com_incidence "with lines title 'Incidence (cumulative)' lc rgb 'purple' lw 2" :-
    @gp :- 1 _sol.t incidence_2 "with lines title 'Incidence_2 (diff)' lc rgb 'brown' lw 2" :-

    println("sum: gem incidence: $(sum(gem_incidence))")
    println("sum: compartment incidence_2: $(sum(incidence_2))")
    println("sum: final cumulative (should be same as above): $(sol[5, end])")
    println("recovered population, should roughly match final cumulative: $(sol[4, end])")
    println("I + E population (+ recovered should be incidence): $(sol[2, end] + sol[3, end])")
    return _sol
end
