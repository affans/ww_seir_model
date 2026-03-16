#import "@preview/typsidian:0.0.3": *
#show: typsidian.with(
  //theme: "dark",
  title: "Notes",
  course: "My Course",
  show-index: false,
  index-entry-list: (
    (
      key: "example",
      short: "An example index entry"
    ),
  )
)

#let rt(txt) = { text(red)[#txt] }
#let bt(txt) = { text(blue)[#txt] }
#let d6(txt) = { text(rgb(0, 180, 255))[#txt] }

Project Details

Idea: the presence of COVID 19 and other pathogens provide protection, through a hypothesis called "The Interferon Mechanism": 


This hypothesis states:  Infection with a respiratory virus (like Influenza) triggers the release of interferons (IFNs) in the host's respiratory mucosa. These interferons create a generalized "antiviral state" in local cells, inhibiting the replication of other viruses. This protection is short-lived but potent, potentially lasting for a few weeks after the initial infection.

Paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC12631739/
Observed data: The winter of 2024/2025 provided a natural experiment for this hypothesis. In both the United States and Europe, a robust Influenza epidemic was observed. WBE data showed a massive surge in Influenza A viral loads in December 2024. Concurrently, SARS-CoV-2 viral loads, which were expected to rise due to the emergence of the transmissible NB.1.8.1 variant, remained surprisingly suppressed or plateaued. It was only in Spring 2025, as the Influenza wave subsided (and the population-level interferon response waned), that SARS-CoV-2 viral loads began to climb significantly


Statistical Evidence for the hypothesis: Statistical analysis using Kendall's Tau ($tau$) confirmed this negative interaction. While Influenza A, B, and RSV showed positive cross-correlations (suggesting environmental factors drive them together), SARS-CoV-2 showed no significant positive correlation with the others, supporting the hypothesis that its dynamics were being actively decoupled or suppressed by the dominant flu wave. [what's the reference for this? see Gem]

== Scientific Question 2: 
=== Quantifying the "Interference Coefficient" in Multi-Pathogen Dynamics

Reference (see conclusion of) https://pmc.ncbi.nlm.nih.gov/articles/PMC12631739/

Gap I am trying to solve:  The "interferon hypothesis" (e.g., Flu suppresses COVID) is gaining observational support, it lacks rigorous quantification. #bt[There is no standard metric to describe the strength of this interaction.  Is the suppression absolute? Is it threshold-dependent?]

=== Methods and Model

Previous models have been statistical. Evidence does exist (e.g., Kendall's Tau paper, the Bengaluro?). 

Here is what we are going to do: 
We hypothesize that the mechanism of interference is innate immune activation (interferon response). I will model this using similar dynamics to when someone is vaccinated. Basically, if you get flu, you are "vaccinated" against COVID for a short period of time. This makes sense because the interferon response provides temporary protection against other viruses. #bt[High prevalence of Influenza induces a temporary "antiviral state" in the population, effectively reducing the susceptibility of individuals to SARS-CoV-2.]

The model
The State Variables:$S$: 
- Susceptible population (shared or separate depending on assumption).
- $I_("flu"), I_("cov")$: Infectious individuals contributing to transmission.
- $P_{"flu"}$: Prevalence of Influenza (driving the interferon "shield").

The general idea is to use this interaction: 
$
  (d I_(c o v))/(d t) = underbrace(beta_(c o v) S I_(c o v), "Normal Transmission") times underbrace((1 - sigma (I_(f l u) (t))/N), "Interference Term") - gamma I_(c o v)
$
Think about how the term $1 - sigma (I_(f l u) (t))/N$ behaves. If $sigma$ is zero, there is no interference and COVID spreads normally. If $sigma$ is positive, then as the prevalence of flu increases, the effective transmission rate of COVID decreases (just like how when someone is vaccinated, their beta is affected). If $sigma$ is 1, then at high flu prevalence, COVID transmission could be completely suppressed.

We don't know what $sigma$ is. *GOAL* The objective is to estimate the posterior distribution of $sigma$ using WW data and Bayesian analysis.  The scientific value is to establish a value for $sigma$ would allow modelers to predict "conditional epidemics." For example, "If the Flu season reaches intensity $X$, the COVID wave will be suppressed by $Y%$."

*How*? Remember, our SEIR model will give us $I(t)$ but we only have wastewater data. What is the relationship between $I(t)$ and the viral load in wastewater? We can use a simple shedding model:
$
  C_(c o v) (t) = k integral_0^infinity I_(c o v) (t - tau) w_(c o v) (tau) d tau + epsilon.alt
$
where $k$ is some scaling factor (accounting for population size, sewer flow, and lab recovery efficiency), $w_(c o v) (tau)$ is the shedding distribution (how much virus an infected person sheds over time), and $epsilon.alt$ is some noise term.

Likelihood function:Wastewater data is highly stochastic (noisy) and over-dispersed (variance > mean) [reference needed]. We do not use a Normal distribution for the error. Instead, we assume the observed viral counts $y_t$ follow a Negative Binomial distribution :
$
y_t ~ "NegBinomial" (mu = C(t), phi.alt)
$
where $mu$ is the model predicted concentration of WW (using the convolution of $I(t)$ with the shedding distribution), and $phi.alt$ is the overdispersion parameter (which we will estimate as well). 

Why Bayesian? We cannot simply "calculate" $sigma$. We must find the value of $sigma$ that makes the observed 2024/2025 wastewater data most probable.

=== Defining the System 
Let's consider three viruses: Influenza, RSV, and SARS-CoV-2. We basically have a SEIR model for each of them separtely, connected by a coupled force of infection. 

#bt[This framework bridges three distinct layers: the biological transmission (Mechanistic), the physical shedding (Observation), and the statistical fitting (Probabilistic).]

#bt[A person can be susceptible to COVID-19 even if they have recovered from the Flu. Therefore, we track the population's status for each virus independently. The interaction occurs only in the Force of Infection.]

The system has 12 state variables 
$ 
  S_v, E_v, I_v, R_v quad "for" v in {C, F, R}
$
where $v$ represents the virus (C for COVID, F for Flu, R for RSV). The compartments $S_v, E_v, I_v, R_v$ represent the susceptible, exposed, infectious, and recovered populations for each virus respectively. For each virus, the transmission dynamics are governed by the following equations:
$
  frac(d S_(v), d t) & = - lambda_(v)(t) S_(v) \ frac(d E_(v), d t) & = lambda_(v)(t) S_(v) - sigma_(v) E_(v) \ frac(d I_(v), d t) & = sigma_(v) E_(v) - gamma_(v) I_(v) \ frac(d R_(v), d t) & = gamma_(v) I_(v)
$
The parameters here are $sigma_v$ which is the latent rate (so $1/sigma_v$ is the latent period). And $gamma_v$ is the recovery rate (so $1/gamma_v$ is the infectious period).

The coupling of these independent group of 4 SEIR equations is through the Force of Infection $lambda_v(t)$, which is defined as:
$
lambda_(v)(t) = underbrace(beta_(v) frac(I_(v), N), "Base Risk") times underbrace(exp (- sum_(k != v) theta_(k -> v) frac(I_(k), N)),("Interference Modifier"))
$
where $beta_v$ is the transmission rate of virus $v$ in a fully susceptible population, and $theta_(k -> v)$ is the interference coefficient that quantifies how much the presence of virus $k$ reduces the transmission of virus $v$. For example, $theta_(F -> C)$ would represent how much the presence of Flu reduces the transmission of COVID. 

Initially I use a shared $theta$: Gem states: A shared $theta$ assumes Symmetric Interference. Biologically, heterologous viral interference is primarily driven by the innate immune system, specifically the release of interferons. When a respiratory virus infects your airway, your cells release an "interferon alarm" that puts the surrounding tissue into a generalized antiviral state. This defense mechanism is non-specific—it blocks COVID just as well as it blocks RSV. Therefore, assuming a single, shared $theta$ is biologically sound and practically solves the "identifiability" problem , as the model only has to find one universal strength for the human interferon response rather than six different directional interaction strengths.

Under the shared hypothesis, the Force of Infection simplifies to:
$
  lambda_(C)(t) &= beta_(C) frac(I_(C), N) exp (- theta frac(I_(F) + I_(R), N)) \ 
  lambda_(F)(t) &= beta_(F) frac(I_(F), N) exp (- theta frac(I_(C) + I_(R), N)) \ 
  lambda_(R)(t) &= beta_(R) frac(I_(R), N) exp (- theta frac(I_(C) + I_(F), N)) \ 
$

*Note* $theta$ (theta) represents the strength of suppression that one virus exerts on another. 
- If $theta = 0$: The term becomes $exp(0) = 1$. The presence of competing viruses has zero effect on transmission.
- If $theta > 0$: The term drops below 1. For example, if $theta = 5$ and $2%$ of the population currently has the Flu ($frac(I_{"flu"}, N) = 0.02$), the multiplier for COVID becomes $exp(-5 times 0.02) = exp(-0.10) approx 0.90$. This means COVID transmission is suppressed by $10%$ due to the ongoing Flu wave.



Note that we won't observe $I_v(t)$ directly. The model will produce such a curve, and we'll calibrate the model using viral load. We observe the viral load in wastewater, which is a convolution of the infectious population with a shedding function. The convolution is 
$
  W_(v)(t) = M_(v) integral_(0)^(infinity) underbrace((sigma_(v) E_(v)(t - tau)), "New Infections at" t - tau) dot underbrace(w_(v)(tau),"Shedding Profile") d tau
$
where $M_v$ is some scaling factor (not sure how this is used??). Gem says #bt[A scalar that converts "number of infected people" into "genome copies per liter". This handles population size, flow rate, and lab efficiency.]. The term is $w_v (tau)$ is the shedding kernel, the probability distribution of viral shedding over time since infection.
#bt[Based on literature (e.g.), we can fix these distributions (typically Gamma densities):
- SARS-CoV-2: Gamma($alpha=2.5, theta=1.8$). Peak ~4.5 days. Long fecal tail.
- Influenza: Gamma($alpha=2.5, theta=1.0$). Peak ~2.5 days.  Short duration.
- RSV: Gamma($alpha=2.0, theta=3.0$). Peak ~6.0 days. Broader shedding window
]

The discrete version of the convolution is:
$
  W_(v)(t) = M_(v) sum_(tau = 0)^(2 1) (sigma_(v) E_(v)(t - tau)) w_(v)(tau)
$
where $M_v$ is the scaling factor, and $(sigma_(v) E_(v)(t - tau))$ are the new infections. 

== Bayesian Analysis. 

We are going to use Bayesian inference to work backward from the Data ($Y$) (the viral load in wastewater) to the Parameters ($theta, beta$). In particular, we'll generate a posterior distribution for $theta$ (the interference coefficient) to quantify the strength of viral interference. Actually even beta and scaling factors in the convolution are unknown, so we will generate a joint posterior distribution for all parameters.

The posterior distribution will answer the hypothesis test above. 

*Details* 
- For $theta$, we can use a half-normal distribution for the prior. This tells the model: "We believe interference is likely zero or positive (suppressive)." 
- For beta, We know from clinical studies that respiratory viruses do not have an $R_0$ of 50. We assign a Log-Normal prior centered around biologically plausible basic reproduction numbers (e.g., $R_0$ between 1.5 and 3.0).

The likelihood: 
The model takes a "guess" at the parameters $Phi$ and runs the ODE to generate a theoretical, perfectly smooth wastewater curve, let's call it $W(t)$. But real wastewater data ($Y$) is messy. It is highly overdispersed due to uneven fecal shedding, toilet flushing dynamics, and PCR testing noise. Therefore, we cannot just calculate a simple "error" or distance between the smooth curve and the data points. Instead, we use a Negative Binomial Likelihood function. The model asks:
"If the true concentration in the sewer was exactly $W(t)$, what is the statistical probability of the lab drawing a sample and getting the specific noisy count $Y(t)$ that is in your CSV file?"$Y(t) ~ "NegativeBinomial"("mean" = W(t), "dispersion" = phi)$. The closer the ODE's predicted curve $W(t)$ is to the actual data points, the higher the Likelihood score.

Here is what the MCMC algorithm does when you press "run":
- It picks a random starting value for $theta$ and $beta$ based on your Priors.
- It runs the ODE and the Shedding Convolution.It calculates the Likelihood score against your real data.
- It then takes a "step" in the mathematical space, tweaking $theta$ slightly (e.g., from 0.1 to 0.15) and runs the whole process again.
- If the new Likelihood score is better, it moves in that direction. If it is worse, it occasionally accepts it anyway to ensure it explores all possibilities and doesn't get stuck.
- It does this thousands of times.

We fit the theoretical concentration $W_v(t)$ to the highly stochastic observed wastewater data $Y_v(t)$ using a Negative Binomial distribution:
$ Y_v (t) ~ {"NegativeBinomial"("mean" = W_v (t), "dispersion" = phi) $

The data itself is distributed using a Negative Binomial (see above). This is because wastewater data is over-dispersed (noisy).

Under the shared $theta$ assumption, we have the following NULL and ALTERNATIVE hypotheses: 

*The NULL Hypothesis* then comes $theta = 0$. (Viruses do not interact; dips in COVID during Flu season are purely coincidental).

*The ALTERNATIVE Hypothesis* is $theta > 0$. (Flu induces a temporary antiviral state that suppresses COVID transmission). If the MCMC algorithm returns a posterior distribution for $theta$ where the entire 95% Credible Interval is strictly greater than 0 (e.g., $95% "CI" = [2.5, 6.1]$), then you have statistically proven your hypothesis. It means there is a 95% probability that biological interference is actively suppressing transmission in your sewershed.

== Parameter Space 
To ensure the model is mathematically solvable, we fix biological constants (latent period $sigma$, recovery period $gamma$, and shedding kernels $w$) based on established clinical literature. The following parameters are explicitly put through Bayesian inference (the MCMC will estimate them):
- The Transmission Rates ($beta_C, beta_F, beta_R$): The intrinsic contagiousness of each virus.
- The Shared Interference Coefficient ($theta$): The core scientific question. It measures the strength of the universal, non-specific innate immune response (interferon shield) triggered by an active respiratory infection.
- Initial Infectious Seeds ($I_{0,C}, I_{0,F}, I_{0,R}$): The number of infected individuals on Day 1 of the simulation.
- Scaling Factors ($M_C, M_F, M_R$): The constants that map "number of infected people" to "viral RNA genome copies per Liter".Dispersion ($phi$): The noise parameter characterizing the variance in the wastewater testing process.

== Problem with Sampling $theta$ 
In our Bayesian analysis, we are sampling $theta$ from a $[0, infinity]$ distribution (e.g., Log-Normal). Remember $theta = 0$ means no interference, and as $theta$ increases, the strength of interference increases. However, the MCMC sampler will struggle to sample $theta = 0$. In fact, The core issue is this: a continuous distribution assigns exactly zero probability mass to any single point. With `Truncated(Normal(0, 5), 0, Inf)`, the probability that the posterior takes the value precisely $theta = 0$ is zero by construction. The MCMC posterior will never have a mean of zero, and asking "is the posterior mass near zero?" is not the same as testing the null $theta = 0$. 

The theoretical solution to this is something called the Spike-and-Slab Prior. This is a mixture of a point mass at zero (the "spike") and a continuous distribution over the positive reals (the "slab"). Replace the prior on $theta$ with a mixture
$
p (theta) = pi_(0) dot delta (0) + (1 - pi_(0)) dot "HalfNormal" (sigma)
$
where $delta(0)$ is a point mass at zero and $pi_0$ is the prior probability of no interference. After sampling, the posterior weight on $delta(0)$ directly gives you $P(theta = 0 | "data")$. This is the most principled approach and lets you compute a Bayes factor for interference vs. no interference. However, this is very difficult to implement in practice, and HMC/NUTS may have a hard time solving this.

Solution: 
Instead of trying to look at the posterior mean of a single model, the gold standard is Bayesian Model Comparison. You fit two separate models and compare their predictive power.
- Model 1 (The Null Model): Hardcode $theta = 0$ in your ODE function. Do not estimate it. This model assumes the viruses circulate entirely independently.
- Model 2 (The Interference Model): Estimate $theta > 0$ using your truncated prior.

Once both models have finished sampling, you compare them using Leave-One-Out Cross-Validation (LOO-CV) or the Watanabe-Akaike Information Criterion (WAIC) (available in Julia via the ParetoSmooth.jl or ArviZ.jl packages). If Model 2 has a significantly better LOO-CV score, you have evidence for viral interference. If the scores are indistinguishable, the "interference" is likely just noise, and you fail to reject the null hypothesis.

Alternatively, we can also compute the Bayes Factor $ B F_(1 0) = p ("data" divides M_(1)) / p ("data" divides M_(0)) $ using the Savage-Dickey density ratio, which compares the posterior density at $theta = 0$ to the prior density at $theta = 0$. However, this is more complex and less commonly used than LOO-CV for model comparison.  

== Units 
The ODE output is infected individuals per day, while wastewater data is weekly viral load in genome copies per liter. We should NOT sum the ODE output over a week to match the weekly data. This will destroy the biological meaning of the convolution. 
- Keep the ODE and Convolution Daily: Your ODE must run on a daily time step to output the number of new infections per day. You then mathematically convolve this daily incidence with your 21-day shedding kernel. The output of this convolution is a continuous, daily curve of the expected wastewater concentration.
- Average the Simulated Wastewater Curve: Because your observed data is a weekly average, you must aggregate your simulated daily wastewater concentrations to match it. For every epidemiological week in your dataset, you extract the 7 simulated daily concentration values from that specific week and calculate their mean.
- The Likelihood Comparison: You then use this simulated weekly average as the mean parameter ($mu$) in your Negative Binomial likelihood function, comparing it directly against your observed w_avg data point.

The scaling factor $M_v$ in the convolution handles this unit conversion, accounting for population size, sewer flow, and lab efficiency.