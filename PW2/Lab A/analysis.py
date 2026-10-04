"""
PW2 Lab A -- Motion from tracking data.

Read noisy free-fall position measurements, then:
  - differentiate once  -> velocity
  - differentiate twice -> acceleration (should be ~ constant -g, but noisy!)
  - integrate the acceleration back up -> recover velocity and position

Complete the TODOs. Run:  python analysis.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

# TODO 1: read freefall.csv into arrays t and y
#         (hint: np.loadtxt with a comma delimiter, skipping the header)
t,y=np.loadtxt("freefall.csv",  delimiter=",", skiprows=1, unpack=True)

# TODO 2: compute velocity v = derivative of y w.r.t. t   (np.gradient)
#         and acceleration a = derivative of v w.r.t. t    (np.gradient again)
#         Print the mean acceleration. Is it close to -9.81? Is it noisy?
v=np.gradient(y,t)
a=np.gradient(v,t)
mean_a = np.mean(a)
print("Mean acceleration:", mean_a)
std_a=np.std(a)
print("Standart deviation:",std_a)

# TODO 3: integrate a back up to recover velocity and position
#         (hint: cumulative_trapezoid(a, t, initial=0) + v[0], then again)


y_original=y.copy()

v=cumulative_trapezoid(a, t, initial=0) + v[0]
y=cumulative_trapezoid(v, t, initial=0) + y_original[0]

diff = np.abs(y - y_original)
max_diff = np.max(diff)
print("Max difference (recovered vs original position):", max_diff)

# TODO 4: make a figure with 3 stacked panels: position, velocity, acceleration
#         vs time. Mark the true -9.81 line on the acceleration panel.
#         Save it as motion.png


fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

axes[0].plot(t, y)
axes[0].set_ylabel("Position (m)")
axes[0].set_title("Reconstructed Free-Fall Motion")

axes[1].plot(t, v)
axes[1].set_ylabel("Velocity (m/s)")

axes[2].plot(t, a)
axes[2].axhline(-9.81, color="red", linestyle="--", label="true g = -9.81 m/s²")
axes[2].axhline(mean_a, color="orange", linestyle="--", label="real a" )
axes[2].set_ylabel("Acceleration (m/s²)")
axes[2].set_xlabel("Time (s)")
axes[2].legend()

plt.tight_layout()
plt.savefig("motion.png")
plt.show()
