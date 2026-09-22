# CSPC - Computer Science for Physics and Chemistry

My coursework repository. Each practical is under PW<n>/Lab <X>/.

## Setup

Create the environment for a given lab:

conda env create -f PW<n>/Lab\ <X>/environment.yml
conda activate cspc

---

## PW1 - Lab A: Reproducible Foundations

**What I built:**
-  A conda environment,a Git repository with a full branch and merge workflow,a test suite for the radioactive decay simulation and a speed benchmark comparing pure-Python and NumPy implementations.

**Speed comparison (loop vs NumPy):**
- loop : 1.623 s
- numpy : 0.0003 s
- speed-up: 5760.4x faster

**Tests:** all passing? (yes / no)
yes 
**Conclusion:**
- The vectorized NumPy version was faster than the pure-Python loop-about 5760x on 200000 atoms—confirming that vectorization matters at scale. All three tests pass,including the check that the simulation's average matches the analytical decay law N0*exp(-lam*t) within tolerance.