# CSPC. Computer Science for Physics and Chemistry

I store my coursework in this CSPC repository.Each practical file is saved under PW<n>/Lab <X>/.

## Setup

Create the environment for a given lab:

```bash

conda env create -f PW<n>/Lab\ <X>/environment.yml

conda activate cspc

```

---

## PW1.Lab A:Reproducible Foundations

**What I built:**

-A conda environment,a Git repository that uses a branch and merge workflow,a test suite for the radioactive decay simulation and a speed benchmark that compares a pure-Python loop with a NumPy usage.

**Speed comparison (loop vs NumPy):**

loop:1.623 s

numpy:0.0003 s

speed-up:5760.4x

**Tests:** all passing? (yes/no)

yes

****

-The vectorized NumPy version was way faster than the pure-Python loop by 5760 times when running 200000 atoms,which shows that vectorization does matter when the problem is big.

  -All three tests passed and the test that confirms the simulation average matches the analytical decay law N0*exp(-lam*t) within tolerance passed as well.

## PW1.Lab B:Data,Plotting and Automation

**What I built:**

-A script (plot.py) that reads the observed decay data from decay_observed.csv into two arrays (t and observed) takes N0 as the observed count builds an analytical curve N0*exp(-lambda*t) and creates a 1x2 figure that compares the observed data (scatter) to the analytical law (line) on shared axes.

-I automated the pipeline with a Snakefile so that running `snakemake --cores 1 figure.png` regenerates the figure only when the CSV file or the script have changed.

**Result:**

-The observed data follows the decay curve very closely.Both panels show the downward exponential shape, which proves that the data matches the theoretical N0*exp(-lambda*t) law.

**Snakemake pipeline:**

-The Snakefile defines one rule that runs plot.py to turn decay_observed.csv into figure.png.Snakemake reruns the rule when the output is missing or older than its input so it would not do anything if run the same code with no changes.

****
-Using np.loadtxt to read the CSV and plotting both curves side by side helped make the comparison simpler.

-Automating the figure generation with Snakemake means the plot always stays in sync with the data and I do not need to rerun the script each time again and again.

## PW2 --- Lab B

### Part 2. Three routes to a minimum

On the convex f(x)=(x-3)^2 + 1 all three methods give x = 3.00000 when x0 = 0.

On g(x)=x^4. 3X^2 + x + 5 the methods dont always agree :

-Beginning from x0=0, gradient descent (x=-1.30084) and SLSQP (x=-1.30086) reach the minimum g=1.486.

Newton converges to x=0.16994,where g''=-5.653<0 so Newton found a maximum,not a minimum.

Starting from x0=2,descent and Newton reach the local minimum x=1.13090 (g=3.930 g''=9.347>0).

SLSQP however ends at the minimum x=-1.30064,jumping over the hill.

Newton solves g'(x)=0 so it can return a maximum;the sign of g'' must be checked.On a -convex function both the starting point and the algorithm decide which minimum is found.

### Part 3. Reaction rate

I fitted a first-order rate k = 0.2618,which is close to the expected 0.25.
The fitted curve C0*exp(-kt) passes through the measurements shown in kinetics.png.

### Part 4. Chemical equilibrium (H2 + I2 <=> 2 HI)

Newton (root‑finding) and SLSQP (minimising k_imbalance(x)^2) agree that the value of x is 0.6595.

The equilibrium composition is H2=0.341 mol,I2=0.341 mol,HI=1.319 mol.

Plugging these values back gives an equilibrium K of about 15 so the equilibrium condition holds.
The plot equilibrium.png shows the reactants falling and HI rising as x increases.

### Part 5 (bonus). Titration equivalence point

I computed the slope dpH/dV with np.gradient.Found its maximum with np.argmax.

The equivalence point is,at V = 50.00 mL,where the maximum slope equals 4.00.

The plot titration.png shows the pH jump and the slope peaking at 50 mL.
