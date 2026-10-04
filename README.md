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

Newton solves g'(x) = 0, so it can return a maximum; the sign of g'' must be
checked. On a non-convex function both the starting point and the algorithm
decide which minimum is found.

### Part 3 — Reaction rate

Fitted first-order rate constant: k = 0.2618 (expected about 0.25). The fitted
curve C0*exp(-kt) passes through the noisy measurements (kinetics.png).

### Part 4 — Chemical equilibrium (H2 + I2 <=> 2 HI)

Newton (root-finding) and SLSQP (minimising k_imbalance(x)^2) agree: x = 0.6595.
Equilibrium composition: H2 = 0.341 mol, I2 = 0.341 mol, HI = 1.319 mol.
Plugging these back gives K of about 15, so the equilibrium condition holds.
The plot (equilibrium.png) shows reactants falling and HI rising with x.

### Part 5 (bonus) — Titration equivalence point

The slope dpH/dV was computed with np.gradient and its maximum found with
np.argmax. The equivalence point is at V = 50.00 mL (max slope = 4.00).
titration.png shows the sharp pH jump and the slope peaking at 50 mL.
