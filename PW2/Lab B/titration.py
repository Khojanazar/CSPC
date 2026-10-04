import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("titration.csv", delimiter=",", skiprows=1)
V, pH = data[:, 0], data[:, 1]

slope = np.gradient(pH, V)
i = np.argmax(slope)
print(f"Equivalence point: V = {V[i]:.2f} mL (max slope = {slope[i]:.2f})")

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(V, pH); ax[0].axvline(V[i], color="r", ls="--")
ax[0].set_xlabel("V base (mL)"); ax[0].set_ylabel("pH")
ax[1].plot(V, slope); ax[1].axvline(V[i], color="r", ls="--")
ax[1].set_xlabel("V base (mL)"); ax[1].set_ylabel("dpH/dV")
plt.tight_layout()
plt.savefig("titration.png", dpi=150)