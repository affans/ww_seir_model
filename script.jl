using DataFrames, CSV
using TidierData
using Random, StatsBase, Distributions
using DifferentialEquations 
using Turing, LogDensityProblems
using SavitzkyGolay # filter
using Gnuplot
using Dates
Gnuplot.options.term = "sixelgd truecolor enhanced size 1000,400 scroll font arial 10"


function make_incidence(X) 
    y = similar(X)
    y .= X - circshift(X, 1)    
    y[1] = y[2]
    return y
end

# helper function to apply the savitzky-golay in the TidierData pipeline.
apply_sg(vec) = savitzky_golay(vec, 7, 1).y
function read_canada() 
    # see data here: https://health-infobase.canada.ca/wastewater/
    # missing values in data denote AGGREGATTIONS
    # process data correctly.
    # the 24/25 season is week 35 of 2024 to week 34 of 2025.

    # we get data for week 26 of 2024, because we use the earlier weeks 
    # to establish the baseline and initial conditions for the model.

    df = DataFrame(CSV.File("canada_wastewater.csv"))
    # clean up a little 
    df = @chain df begin
        # general cleanup 
        @filter(!ismissing(site))
        @mutate(weekstart = Date(weekstart, "mm/dd/yyyy"))
        
        # specific to the project 
        @filter(city == "Regina")
        #@filter(site == "London Greenway North")
        @filter(measureid in ("fluA", "fluB", "covN2", "rsv"))
        # see note above for why epiweek 26
        @filter(EpiYear == 2024 && EpiWeek >= 26 || EpiYear == 2025 && EpiWeek <= 34) 
        @select(pruid, site, city, weekstart, EpiYear, EpiWeek, measureid, w_avg, populationcoverage)
        @pivot_wider(df, names_from = measureid, values_from = w_avg)
        @mutate(flu = fluA + fluB) # combine fluA and fluB into one signal.
        @mutate(cov_sg = ~apply_sg(covN2)) # the ~ means not to vectorize. 
        @mutate(flu_sg = ~apply_sg(flu)) # the ~ means not to vectorize. 
        @mutate(rsv_sg = ~apply_sg(rsv)) # the ~ means not to vectorize. 
    end

    println("length of wastewater data for selected year: $(nrow(df))")
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
    #@gp 
    #@mutate(df, covN2_sg = $covN2_sg)
    return df
end

function sym_defn_ode(du, u, p, t)
    # definitions of SEIR models for each covid, flu, rsv virus 
    # parameters are p = [βᵢ, βᵥ, βᵣ, θ, N] for now (update as needed).
    # the theta is the cross-immunity factor, and N is the total population. 

    # to do: eventually move to params
    σᵥ = 1/3.0  
    γᵥ = 1/5.0  
    σᵢ = 1/2.0
    γᵢ = 1/4.0
    σᵣ = 1/4.0
    γᵣ = 1/7.0

    # v: covid, i: influenza, r: rsv
    Sᵢ, Eᵢ, Iᵢ, Rᵢ, 
    Sᵥ, Eᵥ, Iᵥ, Rᵥ,
    Sᵣ, Eᵣ, Iᵣ, Rᵣ = u

    # unpack the parameters, fourth one is theta. 
    βᵢ, βᵥ, βᵣ, θ, N = p
    
    # calculate the forcce of infection 
    sᵢ = exp(-θ * (Iᵥ + Iᵣ) / N)
    λᵢ = βᵢ * (Iᵢ / N) * sᵢ

    sᵥ = exp(-θ * (Iᵢ + Iᵣ) / N)
    λᵥ = βᵥ * (Iᵥ / N) * sᵥ

    sᵣ = exp(-θ * (Iᵢ + Iᵥ) / N)
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
const K_COV = make_kernel(2.5, 1.8, 21) # Peak ~4.5 days, long tail (fecal)
const K_FLU = make_kernel(2.5, 1.0, 21) # Peak ~2.5 days, sharp drop (respiratory)
const K_RSV = make_kernel(2.0, 2.5, 21) # Peak ~5.0 days, broader (pediatric shedding)

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
        pop_size,
        I_flu_0 = 2.0,
        I_cov_0 = 500.0, 
        I_rsv_0 = 2.0,
        I_cov_flag = false # flag 

    )
    # obs_ww_flu, obs_ww_cov, obs_ww_rsv are the 
    # observed wastewater signals for each virus,
    # which we will compare to the simulated wastewater signal from the ODE.
    
    # general idea is to sample the relevant parameters (like beta and theta) 
    # and then run the ODE to get the simulated incidence
    # The simulated incidence is then convolved with the shedding kernel 
    # to get the simulated wastewater signal, 
    # which is used as the mean of a likelihood function (Negative Binomial) to compare to the observed wastewater data.

    # parameters that will be sampled: theta, beta for each virus, scaling factors for each virus, and dispersion parameter for the likelihood.
    
    # theta - cross immunity factor. 0 means no cross-immunity, higher means more.
    # Q: should we make this go negative as well? 
    # because θ = 0 is the null hypothesis, we want to allow the sampler to explore values close to 0.
    # if it's always truncated like this, then maybe all credible intervals will be 
    # entirely above 0, which might be misleading.
    # See written notes for details on the variance of this prior and how it affects the inference. 
    θ ~ Truncated(Normal(0, 5.0), 0, Inf) 

    # transmission parameters for each virus. 
    βᵢ ~ Truncated(LogNormal(log(0.1), 1.0), 0.0, 1.5) 
    βᵥ ~ Truncated(LogNormal(log(0.1), 1.0), 0.0, 1.5) 
    βᵣ ~ Truncated(LogNormal(log(0.1), 1.0), 0.0, 1.5) 

    # scaling factors to map the ODE's I(t) to the observed wastewater signal though convolution
    # for LogNormal(0, 2.0) this is extremely wide, spans [0.018, 54.6] — sampler wastes time at M=30
    # see Obsidian notes
    M_c ~ LogNormal(0, 0.5)
    M_f ~ LogNormal(0, 0.5)
    M_r ~ LogNormal(0, 0.5)

    #  exponential is s Numerically Fragile (can sample too close to 0) which in the division below causes issue (since phi is the dispersion in the NB).  A more robust prior is:
    # phi ~ Exponential(1.0) # the dispersion parameter for the Negative Binomial likelihood.
    phi ~ Gamma(2.0, 1.0)   # mean=2, bounded away from zero

    # setup the ODE for running 
    # get the timing
    weeks = length(obs_ww_flu) # or obs_ww_cov, obs_ww_rsv, they should all be the same length.
    days = weeks * 7  
    tspan = (0.0, Float64(days))
    params = [βᵢ, βᵥ, βᵣ, θ, pop_size] # params vector: [βᵢ, βᵥ, βᵣ, θ, N] 

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

    # if bayesian_initial_cond 
    #     Iᵥ ~ Truncated(Normal(500, 300), 1.0, pop_size * 0.05)
    #     initial_cond[7] = round(Int64, Iᵥ) # see note above about rounding and dual numbers.
    # end

    _I_cov_0 = I_cov_0  # initialise with default, stays as Float64 for AD
    if I_cov_flag # Iᵥ stays as a Dual number — no rounding, no Int conversion, the ODE solver accepts Float64 and Dual numbers equally
        _I_cov_0 ~ Truncated(Normal(500.0, 300.0), 1.0, pop_size * 0.05)
    end
    
    # Construct u₀ fresh each sample — no mutation of external arrays
    # E=0, R=0 everywhere given new-strain assumption
    initial_cond = [
        pop_size - I_flu_0, 0.0, I_flu_0, 0.0,   # flu:   S E I R
        pop_size - _I_cov_0, 0.0, _I_cov_0, 0.0,   # covid: S E I R
        pop_size - I_rsv_0, 0.0, I_rsv_0, 0.0,   # rsv:   S E I R
        0.0, 0.0, 0.0                              # incidence
    ]
    
    # solve the ODE 
    prob = ODEProblem(sym_defn_ode, initial_cond, tspan, params)
    _sol = solve(prob, Tsit5(), saveat=1.0)
    sol = reduce(hcat, _sol.u)

    # get the ODE simulated incidence (this will be in days)
    Iᵢ_sim = make_incidence(sol[13, :])
    Iᵥ_sim = make_incidence(sol[14, :])
    Iᵣ_sim = make_incidence(sol[15, :])

    # convert incidfence to wastewater data  
    Wᵢ_sim = sym_convolve_seir(Iᵢ_sim, K_FLU) * M_f
    Wᵥ_sim = sym_convolve_seir(Iᵥ_sim, K_COV) * M_c
    Wᵣ_sim = sym_convolve_seir(Iᵣ_sim, K_RSV) * M_r
    
    #likelyhood, but we'll have to go in weeks 
    for w in 1:weeks 
        idx_end = w * 7
        if idx_end <= length(Wᵢ_sim)

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
            
            p_flu = phi / (phi + mu_i) # convert to rate/dispersion for NB
            if !ismissing(obs_ww_flu[w]) 
                # if the observed ww signal is very low (below 0.03), it is at the LOD 
                # it means the the signal is somewhere between 0 and 0.021. 
                # I think we can just get by with saying it's 0, since the likelihood will be very similar for any value in that range, and it simplifies the math a lot.
                if obs_ww_flu[w] <= 0.03
                    Turing.@addlogprob! logpdf(NegativeBinomial(phi, p_flu), 0)
                else
                    obs_ww_flu[w] ~ NegativeBinomial(phi, p_flu) 
                end
            end

            p_cov = phi / (phi + mu_c) # conver to rate/dispersion
            if !ismissing(obs_ww_cov[w]) 
                obs_ww_cov[w] ~ NegativeBinomial(phi, p_cov) 
            end
            
            p_rsv = phi / (phi + mu_r) # convert to rate/dispersion
            if !ismissing(obs_ww_rsv[w]) 
                if obs_ww_rsv[w] <= 0.09
                   Turing.@addlogprob! logpdf(NegativeBinomial(phi, p_rsv), 0)
                else
                    obs_ww_rsv[w] ~ NegativeBinomial(phi, p_rsv) 
                end
            end
        end
    end
end

function run_inference() 
    N = 285_039  # fix population to Regina
    
    # initial conditions # we need to estimate covid initial conditions from the data 
    # for regina, flu and rsv were recorded as 0.008213384 which is the LOD (limit of detection)
    # (this needs to be acounted for in the inference model).
    Iᵢ = 2 
    Iᵥ = 2 # will be replaced in the function (make sure flag is set -- see function definition)
    Iᵣ = 2

    # what about Recovered people for pre-existing immunity?
    # we also know there was quite a bit of people with prior exposure to COVID 
    # so we should set the Recovered compartment for COVID to be non-zero 
    # at the start of the season, to reflect that prior immunity.
    # but we are assuming that by the 2024 season, the cross-immunity 
    # from that prior exposure has waned, which is why we are not setting Rᵥ(0) to be very high.

    # get WW data 
    df = read_canada() 
    obs_flu = df.flu_sg
    obs_covid = df.cov_sg
    obs_rsv = df.rsv_sg
    @assert length(df.flu_sg) == length(df.cov_sg) == length(df.rsv_sg)

    model = inference_model(obs_flu, obs_covid, obs_rsv, N, Iᵢ, Iᵥ, Iᵣ, true)
    chain = sample(model, NUTS(), MCMCThreads(), 1000, 4)  # 1000 samples, 4 chains for parallel sampling
    return chain
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

    obs_flu = Float64.(rand.(Poisson.(Wᵢ_sim)))
    obs_covid = Float64.(rand.(Poisson.(Wᵥ_sim)))
    obs_rsv =  Float64.(rand.(Poisson.(Wᵣ_sim)))
    
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
