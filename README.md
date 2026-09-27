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