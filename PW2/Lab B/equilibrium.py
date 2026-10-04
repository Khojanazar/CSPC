import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton, minimize

K = 15.0   # ВОЗЬМИ значение K из скелета с Moodle

def k_imbalance(x):
    return (2*x)**2 / ((1 - x)*(1 - x)) - K

x_newton = newton(k_imbalance, 0.5)
x_slsqp = minimize(lambda v: k_imbalance(v[0])**2, [0.5],
                   method="SLSQP", bounds=[(0, 0.999)]).x[0]
print(f"Newton: x = {x_newton:.4f}")
print(f"SLSQP:  x = {x_slsqp:.4f}")

x = x_newton
print(f"H2 = {1-x:.3f} mol, I2 = {1-x:.3f} mol, HI = {2*x:.3f} mol")

xs = np.linspace(0, 0.99, 200)
plt.plot(xs, 1 - xs, label="H2")
plt.plot(xs, 1 - xs, "--", label="I2")
plt.plot(xs, 2 * xs, label="HI")
plt.axvline(x, color="k", ls=":", label=f"equilibrium x = {x:.2f}")
plt.xlabel("extent x"); plt.ylabel("moles"); plt.legend()
plt.savefig("equilibrium.png", dpi=150)