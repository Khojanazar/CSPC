# CSPC - Computer Science for Physics and Chemistry

My coursework repository. Each practical is under PW<n>/Lab <X>/.

## Setup

Create the environment for a given lab:

conda env create -f PW<n>/Lab\ <X>/environment.yml
conda activate cspc

---

## PW1 - Lab A: Reproducible Foundations

**What I built:**
-  A conda environment,a Git repository with a full branch and merge workflow,a test suite for the radioactive decay simulation  and a speed benchmark comparing pure-Python and NumPy implementations.

**Speed comparison (loop vs NumPy):**
- loop : 1.623 s
- numpy : 0.0003 s
- speed-up: 5760.4x faster

**Tests:** all passing? (yes / no)
yes 
**Conclusion:**
- The vectorized NumPy version was faster than the pure-Python loop-about 5760x on 200000 atoms—confirming that vectorization matters at scale. All three tests pass,including the check that the simulation's average matches the analytical decay law N0*exp(-lam*t) within tolerance.


## PW1 - Lab B: Data, Plotting, and Automation

**What I built:**
- A script (plot.py) that reads the observed decay data from decay_observed.csv into
  two arrays (t and observed), computes N0 as the first observed count, builds the
  analytical curve N0*exp(-lambda*t), and produces a 1x2 figure comparing the observed
  data (scatter) to the analytical law (line) on shared axes. Automated the whole
  pipeline with a Snakefile so that running `snakemake --cores 1 figure.png` regenerates
  the figure only when the data or script have changed.

**Result:**
- The observed data closely follows the analytical decay curve — both panels show the
  same downward exponential shape, confirming the data matches the theoretical
  N0*exp(-lambda*t) law reasonably well.

**Snakemake pipeline:**
- The Snakefile defines one rule that runs plot.py to turn decay_observed.csv into
  figure.png; Snakemake only reruns the rule when the output is missing or older than
  its input, so re-running the same command with no changes does nothing.

**Conclusion:**
- Reading the CSV with np.loadtxt, building the analytical curve, and plotting both
  side by side made the comparison straightforward. Automating the figure generation
  with Snakemake means the plot always stays in sync with the data without manually
  rerunning the script every time.

## PW2 --- Lab A

**Results:**
- Mean acceleration: **-8.58 m/s²** — close to the true value of -9.81 m/s², confirming the object is in free fall (the offset is due to measurement noise).
- Standard deviation of acceleration: **28.72 m/s²** — very large and noisy. This happens because acceleration is obtained from *two* successive numerical differentiations of the position data. Each differentiation compares nearby, slightly noisy measurements and divides by a small time step, which amplifies the noise; doing this twice compounds the effect, so even small measurement noise in position becomes huge swings in acceleration.
- Max difference between recovered and original position: **0.78 m** — after integrating the noisy acceleration back up to velocity and then position, the recovered position matches the original within about 0.78 metres. This shows that integration (a cumulative sum) partially cancels out random noise, which is the opposite effect of differentiation — the central lesson of this lab.

**Files:** `analysis.py`, `motion.png`, `freefall.csv`, `trajectory.csv` in `PW2/Lab A/`.








## PW2 --- Lab B


### Part 2 — Three routes to a minimum

On the convex f(x) = (x-3)^2 + 1, all three methods give x = 3.00000 from x0 = 0.

On g(x) = x^4 - 3x^2 + x + 5 the methods do not always agree:
- x0 = 0: gradient descent (x = -1.30084) and SLSQP (x = -1.30086) reach the
  global minimum, g = 1.486. Newton converges to x = 0.16994, where
  g'' = -5.653 < 0, so it found a local maximum, not a minimum.
- x0 = 2: gradient descent and Newton reach the local minimum x = 1.13090
  (g = 3.930, g'' = 9.347 > 0). SLSQP, however, ended at the global minimum
  x = -1.30064, jumping over the hill.

Conclusions: Newton solves g'(x) = 0, so it can return a maximum; the sign of
g'' must be checked. On a non-convex function both the starting point and the
algorithm decide which minimum is found. Only the global minimum (x ≈ -1.30)
has the lowest value of g.


### Part 3 — Reaction rate
Fitted first-order rate constant: k = 0.2618 (expected ≈ 0.25). The fitted
curve C0·exp(-kt) passes through the noisy measurements (kinetics.png).

### Part 4 — Chemical equilibrium (H2 + I2 ⇌ 2 HI)

Starting from 1 mol H2 and 1 mol I2, the equilibrium extent was found in two ways:
root-finding with `scipy.optimize.newton` and minimisation of `k_imbalance(x)**2`
with SLSQP. Both methods agree: x = 0.6595.

Equilibrium composition: H2 = 0.341 mol, I2 = 0.341 mol, HI = 1.319 mol
(HI = 2x ≈ 1.32 mol, consistent with the expected x ≈ 0.66). Plugging these back
into [HI]^2 / ([H2][I2]) gives K ≈ 15, so the equilibrium condition is satisfied.

The plot (equilibrium.png) shows the reactants H2 and I2 decreasing and the
product HI increasing with the extent x; the equilibrium is marked at x = 0.6595.

### Part 5 (bonus) — Titration equivalence point

The slope dpH/dV was computed with `np.gradient(pH, V)`, and the equivalence point
was taken where the slope is largest (`np.argmax`). The equivalence point is at
V = 50.00 mL, where the slope reaches its maximum of 4.00 pH/mL. The left plot
(titration.png) shows the sharp jump of the pH curve, and the right plot shows its
slope peaking at 50 mL.





## PW3 - Data,Distributions,Testing,and Causation

**What I built:**
 This Python script is designed for data analysis. It starts by loading datasets. Then, it looks at how the data is spread out and checks if it follows a normal distribution using statistical tests. After that, it compares different clinical groups with appropriate statistical tests. The script also calculates how different variables relate to each other through correlations and identifies any confounding variables in a dataset that looks at chemical exposure.

### Session 1:Characterising Distributions & Normality

 **Data Inspection**: Loaded `Heart_disease_cleveland_new.csv` (303 rows, 14 columns) and isolated continuous variables: `age`, `chol`, `trestbps`, and `thalach`.

 **Normality Tests (Shapiro-Wilk)**:
`age`:$p=0.00607$(Non-normal)
`chol`:$p=0.00000$(Non-normal)
`trestbps`:$p==0.00000$(Non-normal)
`thalach`: $p=0.00007$(Non-normal)

-**Reasoning**: As all continuous variables failed normality checks ($p < 0.05$),non-parametric statistical methods (such as the Mann-Whitney U test) were chosen for group comparisons in Session 2.

### Session 2:Drawing Conclusions & Causation
 **Heart Rate Comparison (`thalach`)**: 
Disease group mean: $139.26 \pm 1.92$ (SE)
  Healthy group mean: $158.38 \pm 1.50$ (SE)
  Mann-Whitney U test result: $U = 5806.50$, $p = 1.86 \times 10^{-13}$.


  **Conclusion**: Healthy individuals achieve a statistically significant higher maximum heart rate during stress testing compared to patients with heart disease.

 **Age vs. Maximum Heart Rate**: 
  Pearson correlation: $r = -0.3938$, $p = 1.11 \times 10^{-12}$.

  **Interpretation**: Maximum heart rate systematically decreases as age increases, reflecting normal physiological aging.

 **Chemical-Exposure Mystery**:

   *Naive Analysis*: Cadmium showed an extremely strong correlation with malignancy ($r = 0.8800$) compared to Benzene ($r = 0.3847$), making Cadmium look like the root cause at first glance.
   *Controlled Analysis*: After filtering the dataset to control for the confounding variable (`pollution_index` between 40 and 60), the spurious association shifted: Benzene's correlation rose to $r = 0.7016$ while Cadmium dropped to $r = 0.3157$.

   *Conclusion*: **Benzene** is the true chemical driver of tumour malignancy; **Cadmium** only appeared guilty due to confounding by area pollution levels.
 **Bonus (Category Balance)**: Target proportions are well-balanced at ~54.1% healthy (0) and ~45.9% disease (1).