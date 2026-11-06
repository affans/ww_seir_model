## FUNCTIONS FOR REVISION 2 - PAIRWISE SIMULATIONS
using DelimitedFiles, DataFrames

const SIM_SCENARIOS = Dict(
    :null         => zeros(3, 3),                 # false-positive check
    :homogeneous  => fill(12.0, 3, 3),            # recovery check (true θ = 12)
    :covid_driven => [0.0 20.0  4.0;              # flu: strongly suppressed by COVID
                      4.0  0.0  4.0;              # COVID: weakly suppressed
                      4.0 20.0  0.0],             # RSV: strongly suppressed by COVID
    :flu_rsv_only => [0.0  0.0 12.0;              # only flu <-> RSV interact
                      0.0  0.0  0.0;
                      12.0 0.0  0.0],
)

function sym_defn_ode_pairwise(du, u, p, t)
    # if changing these parameters, change in the analysis function also
    σᵢ = 1/2.0; γᵢ = 1/4.0
    σᵥ = 1/3.0; γᵥ = 1/5.0  
    σᵣ = 1/4.0; γᵣ = 1/5.0

    # v: covid, i: influenza, r: rsv
    Sᵢ, Eᵢ, Iᵢ, Rᵢ, 
    Sᵥ, Eᵥ, Iᵥ, Rᵥ,
    Sᵣ, Eᵣ, Iᵣ, Rᵣ = u

    # remove the theta from p.v and send it in separately as p.Θ
    βᵢ_base, βᵢ_peak, βᵥ_base, βᵥ_peak, βᵣ_base, βᵣ_peak, N, ε_flu, ε_cov, ε_rsv, 
        t_school_flu, t_school_cov, t_school_rsv = p.v
    Θ = p.Θ

    βᵢ = βᵢ_base + (βᵢ_peak - βᵢ_base) * logistic((t - t_school_flu) / 7.0)
    βᵥ = βᵥ_base + (βᵥ_peak - βᵥ_base) * logistic((t - t_school_cov) / 7.0)
    βᵣ = βᵣ_base + (βᵣ_peak - βᵣ_base) * logistic((t - t_school_rsv) / 7.0)

    λᵢ = βᵢ * (Iᵢ / N) * exp(-(Θ[1,2] * Iᵥ + Θ[1,3] * Iᵣ) / N)
    λᵥ = βᵥ * (Iᵥ / N) * exp(-(Θ[2,1] * Iᵢ + Θ[2,3] * Iᵣ) / N)
    λᵣ = βᵣ * (Iᵣ / N) * exp(-(Θ[3,1] * Iᵢ + Θ[3,2] * Iᵥ) / N)
    
    # calculate force of infection: 
    λᵢ = βᵢ * (Iᵢ / N) * exp(-(Θ[1,2] * Iᵥ + Θ[1,3] * Iᵣ) / N)
    λᵥ = βᵥ * (Iᵥ / N) * exp(-(Θ[2,1] * Iᵢ + Θ[2,3] * Iᵣ) / N)
    λᵣ = βᵣ * (Iᵣ / N) * exp(-(Θ[3,1] * Iᵢ + Θ[3,2] * Iᵥ) / N)

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

# simulate weekly wastewater from the pairwise model, using posterior means of
# all other parameters from the fitted m1 chain as the "truth"
function simulate_ww(Θ, chain, name; seed = 1)
    Random.seed!(seed)
    df, pp = read_province(name)
    weeks = nrow(df) 
    N = pp.popsize
    i0c = pm(chain, :i0c) 

    # initial conditions for the SEIR model 
    u0 = [(1 - pp.immunity_flu) * N - 1.0, 1.0, 0.0, pp.immunity_flu * N,
          (1 - pp.immunity_cov) * N - i0c, i0c, 0.0, pp.immunity_cov * N,
          (1 - pp.immunity_rsv) * N - 1.0, 1.0, 0.0, pp.immunity_rsv * N,
          0.0, 0.0, 0.0]

    # get the fitted parameters from the chain, and solve the ODEs
    v = vcat([pm(chain, s) for s in (:βᵢ_base, :βᵢ_peak, :βᵥ_base, :βᵥ_peak, :βᵣ_base, :βᵣ_peak)],
             N, pm(chain, :ε_flu), pm(chain, :ε_cov), pm(chain, :ε_rsv),
             pm(chain, :t_sch_flu), pm(chain, :t_sch_cov), pm(chain, :t_sch_rsv))
    
    sol = solve(ODEProblem(sym_defn_ode_pairwise, u0, (0.0, weeks * 7.0), (v = v, Θ = Θ)), Tsit5(), saveat = 1.0)
    
    k = KERNELS[:base] # pick the base kernel for the simulation
    ww(row, kern, logM) = sym_convolve_seir(make_incidence(sol[row, :]) ./ N, kern) .* exp(pm(chain, logM))
    weekly(x) = [max(mean(x[w*7-6:w*7]), 1e-6) for w in 1:weeks]   # same aggregation as the model
    
    # add lognormal noise to the weekly wastewater values, using the fitted σ parameters
    noisy(x, σ) = rand.(LogNormal.(log.(weekly(x)), pm(chain, σ)))
    
    obs = (flu = noisy(ww(13, k.flu, :log_M_f), :sigma_flu),
           cov = noisy(ww(14, k.cov, :log_M_c), :sigma_cov),
           rsv = noisy(ww(15, k.rsv, :log_M_r), :sigma_rsv))
    
           # benchmark: average of Θ weighted by prevalence overlap I_p(t) * I_q(t)
    I = (sol[3, :], sol[7, :], sol[11, :])
    num, den = 0.0, 0.0
    for p in 1:3, q in 1:3
        p == q && continue
        w = sum(I[p] .* I[q]); num += Θ[p, q] * w; den += w
    end
    return obs, num / den
end

function run_simulation_study(prov = "Ontario")
    # read the fitted chain for the province, and simulate data under each scenario
    chain = read_chain("outputs/chain_$(prov)_m1_base.jls")
    _, pp = read_province(prov)
    hp = hyperparameters(prov)
    results = []
    println("running with 2000 iterations, high scale")
    res = pmap(collect(SIM_SCENARIOS)) do (scen, Θ)
        # fname: 
        #  r2rev_$(prov)_r2sensitivity_$(scen).jls -> θ ~ Truncated(Normal(0, 2.0), 0, Inf)
        #  r2rev_$(prov)_high_scale_$(scen).jls -> θ ~ Truncated(Normal(0, 10.0), 0, Inf)
        
        fname = "outputs/r2rev_$(prov)_high_scale_$(scen).jls"
        println("Running scenario $(scen) for province $(prov)...")
        println("saving at: $(fname)")        
        # geenrated data from the ODE model
        obs, θ_bench = simulate_ww(Θ, chain, prov)
        model = inference_model(obs.flu, obs.cov, obs.rsv, hp, pp;  θ_scale = 10.0)   # unchanged single-θ model
        fit = sample(model, NUTS(), MCMCThreads(), 2000, 4)
        save_chain(fname, fit)
        θs = vec(Array(fit[:θ]))
        # push!(results, (scenario = scen, θ_benchmark = round(θ_bench, digits = 2), θ_mean = mean(θs), θ_lo = quantile(θs, 0.025), θ_hi = quantile(θs, 0.975)))
        # println(results[end])
        return (scenario = scen, θ_benchmark = round(θ_bench, digits = 2), θ_mean = mean(θs), θ_lo = quantile(θs, 0.025), θ_hi = quantile(θs, 0.975))
    end
    df = DataFrame(res)
    CSV.write("outputs/r2rev_$(prov)_r2sensitivity.csv", df)
    return df 
end

function practical_ident(prov)
    posterior, df, province_params, hp = redirect_stdout(devnull) do
        posterior = read_chain("outputs/chain_$(prov)_m1_base.jls")
        df, province_params = read_province(prov)
        hp = hyperparameters(prov)

        (posterior, df, province_params, hp)
    end
    #posterior = read_chain("outputs/chain_$(prov)_m1_base.jls")
    #df, province_params = read_province(prov);
    #hp = hyperparameters(prov);
    obs_flu = df.flu_sg
    obs_covid = df.cov_sg
    obs_rsv = df.rsv_sg
    
    model = inference_model(obs_flu, obs_covid, obs_rsv, hp, province_params; nullmodel=false, freetheta=false, kernels = KERNELS[:base])
    prior = sample(model, Prior(), 1000)
    ps = intersect(names(posterior, :parameters), names(prior, :parameters))
    # get the posterior and prior samples for the parameters of interest
    P = Array(posterior[ps]); 
    Q = Array(prior[ps])
    
    
    mP, sP = vec(mean(P; dims=1)), vec(std(P; dims=1))
    mQ, sQ = vec(mean(Q; dims=1)), vec(std(Q; dims=1))
    
    # calculate contraction and shift for each parameter
    contraction = 1 .- (sP ./ sQ).^2 
    shift = (mP .- mQ) ./ sQ 

    # pearson correlation between θ and all other parameters
    C  = cor(P) # correlation matrix of posterior samples

    iθ = findfirst(==(:θ), ps) # find index of the theta
    
    # find the parameter with the highest absolute correlation to θ
    r  = [i == iθ ? 0.0 : abs(C[iθ, i]) for i in eachindex(ps)]
    j  = argmax(r)

    println("""
        θ prior: $(mQ[iθ]), SD: $(sQ[iθ])
        θ posterior: $(mP[iθ]), SD: $(sP[iθ])
        θ shift: $(shift[iθ]), 
        contraction: $(contraction[iθ])
        max |r|: $(r[j]) with parameter: $(ps[j])
    """)

    #other parameters (everything except θ)
    others = setdiff(eachindex(ps), [iθ])

    #find the parameter with the lowest contraction, which is the one the data informed least and where the prior did most of the work.
    k = others[argmin(contraction[others])]

    println("""
        other params median contraction: $(median(contraction[others]))
        lowest contraction: $(contraction[k]) with parameter: $(ps[k])
    """ )

    ## goes through every pair of parameters once (b in a+1:... skips duplicates and self-pairs). It keeps any pair whose posterior correlation is above 0.7 in absolute value. Each kept pair is stored as (name1, name2, r).
    strong = [(ps[a], ps[b], round(C[a,b]; digits=2)) for a in eachindex(ps) for b in a+1:length(ps) if abs(C[a,b]) > 0.7]
    println("   pairs with |r| > 0.7: ", isempty(strong) ? "none" : strong)
   
    return nothing
    
end

function run_diagnostics()
    provs = ["Ontario", "British Columbia", "Alberta", "Manitoba", "Quebec", "Nova Scotia"]
    for prov in provs
        println("running province: $(prov)")
        practical_ident(prov)
    end

end
