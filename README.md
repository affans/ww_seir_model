# WW Bayesian SEIR Model

Code for the paper by Shoukat, Moosavi, Hansmeier, Wu. The code is written in Julia (version: Version 1.10.11 (2026-03-09)) and the repository includes environment files `Project.toml` and `Manifest.toml` which ensures full reproducibility. 



## Repository contents

| File | Purpose |
|---|---|
| `script.jl` | All the code is contained in this one file including data processing, ODE model, MCMC model, inference, post hoc analysis, and plotting. |
| `parameters.toml` | Fixed province-level inputs: effective population size, pre-season immunity for each pathogen, and school-timing centres. |
| `canada_wastewater.csv` | Snapshot of the national wastewater surveillance data from the Public Health Agency of Canada Health Infobase (<https://health-infobase.canada.ca/wastewater/#download>). |
| `covid19-wastewater-data-dictionary-en.csv` | Data dictionary for the CSV above. |
| `Project.toml` | Julia package dependencies. |

The `outputs/` folder is not tracked by git, but needs to be available in the cloned repository to save the MCMC chains, the results log, and the figures. Alternatively, adjust the paths as needed. 

The MCMC inference runs 4 chains in parallel with `MCMCThreads()` by default. Start Julia with at least 4 threads (`julia -t 4`); with fewer threads the chains run one after another.

For plotting, `gnuplot` installed on the system and available on your `PATH`. The figures are saved with the `pdfcairo` terminal, so gnuplot must be built with cairo support. Check with `gnuplot -e "set terminal"` and look for `pdfcairo`. **Alternatively**, one may write their own plotting functions. 

## Steps to Reproduce

```bash
git clone https://github.com/affans/ww_seir_model
cd ww_seir_model
mkdir -p outputs/figures
```

Then, in Julia, from the repository folder:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
Pkg.add(["Gnuplot", "GarishPrint"])   # used by script.jl but not yet listed in Project.toml
```

All paths in `script.jl` are relative (`parameters.toml`, `canada_wastewater.csv`, `outputs/...`), so Julia must be started from the repository root.

Load the code:

```julia
include("script.jl")
```


### Running the code for a province

```julia
chain = run_inference("Ontario", :m1)     # model with interference (θ estimated)
chain = run_inference("Ontario", :null)   # null model M0 (θ fixed at 0)
save_chain("outputs/chain_Ontario_m1.jls", chain)
```

`run_inference(name, model)` builds the Turing model with the province's data, its priors (`hyperparameters(name)`), and its fixed inputs (`parameters.toml`). It then samples with NUTS: 4 chains × 4000 draws, Turing's default adaptation settings. It returns an `MCMCChains.Chains` object. It does **not** save the chain; use `save_chain`.

The second argument selects the model:

- `:m1` — the interference model. **All main results in the paper come from this model.**
- `:null` — the same model with `θ = 0`. Used only for the M0 vs M1 incidence comparison (`plot_incidence_curves`) and the M0 section of the results log.

### All provinces

```julia
run_all_scenarios()
```

This fits both `:null` and `:m1` for all six provinces and saves 12 files:

```
outputs/chain_<Province>_m1.jls
outputs/chain_<Province>_null.jls
```

where `<Province>` is the name with spaces, e.g. `outputs/chain_British Columbia_m1.jls`. The chains are saved with Julia's `Serialization`, so they must be read back with the same Julia and package versions (`read_chain(filename)`).

### Post hoc analysis

```julia
run_analysis_all_prov()
```

This reads the saved chains for all six provinces and runs `chain_analysis` on each, for M1 and then M0. All printed output goes to `outputs/full_results.log`. **Both the `_m1` and `_null` chains must exist**, or the function stops with an error.

For each province and model, `chain_analysis(chain, name)`:

1. draws 1000 posterior samples from the chain,
2. solves the ODE forward for each draw,
3. computes posterior incidence, prevalence, predicted wastewater concentration, and the effective reproduction number R_t for each pathogen (with and without interference),
4. reports peak size and peak timing of incidence (mean and 95% credible interval), the posterior of `θ`, and the implied suppression of each pathogen.

`chain_analysis` also returns these quantities as a named tuple, so it can be used interactively:

```julia
res = chain_analysis(read_chain("outputs/chain_Ontario_m1.jls"), "Ontario"; plot_figs = false)
res.meanθ, res.lowθ, res.highθ
```

The province table of `θ` and suppression (Flu, COVID, RSV) is printed in Markdown by:

```julia
plot_theta_and_suppression()
```

###  Figures

Each plotting function reads the saved chains where needed and writes a PDF.

| Function | Output | Chains needed |
|---|---|---|
| `plot_wastewater()` | `outputs/figures/wastewater_raw_allprovs.pdf`, `outputs/figures/wastewater_raw_twoprovs.pdf` | none (data only) |
| `plot_fitted_wastewater_signal()` | `outputs/figures/figure1_ww_fitted_model.pdf`, `outputs/figures/figure1_ww_fitted_model_first3.pdf` | M1 |
| `plot_ridgeplots()` | `outputs/figures/theta_histograms.pdf` | M1 |
| `plot_theta_vs_covid_ranked()` | `outputs/figures/theta_vs_covid_ranked.pdf` | none (see note below) |
| `plot_incidence_curves()` | `outputs/incidence_curves.pdf` | M0 and M1 |
| `plot_shedding_kernels()` | `outputs/shedding_kernels.pdf`, `outputs/shedding_kernels.svg` | none |

Note: `plot_theta_vs_covid_ranked()` uses the `θ` posterior means and intervals from the paper, written directly in the function. If you refit the model, update these values from `outputs/full_results.log` (or `plot_theta_and_suppression()`) before making this figure.

Helper and development functions that are not needed to reproduce the results: `plot_province`, `solve_seir_model` (forward simulation with fixed parameters), and `test_ad` (compares automatic-differentiation back-ends; needs extra packages).

## License

MIT. See `LICENSE`.
