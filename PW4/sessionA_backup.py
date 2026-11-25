import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)

df = pd.read_csv("data/concrete.csv")
print("shape:", df.shape)

# --- чистка ---
n_dup = df.duplicated().sum()
n_nan = df.isna().sum().sum()
print("дубликатов:", n_dup, "| пропусков:", n_nan)

df = df.drop_duplicates().reset_index(drop=True)
print("после чистки:", df.shape)

# пересобираем X и y из очищенной таблицы
X = df.drop(columns="strength")
y = df["strength"]


print(df.describe().T)    

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].scatter(df["cement"], df["strength"], s=8, alpha=0.6)
axes[0].set_xlabel("cement (kg/m3)")
axes[0].set_ylabel("strength (MPa)")
axes[1].scatter(df["age"], df["strength"], s=8, alpha=0.6)
axes[1].set_xscale("log")
axes[1].set_xlabel("age (days, log scale)")
axes[1].set_ylabel("strength (MPa)")
plt.tight_layout()
plt.savefig("figures/A2_scatter.png", dpi=150)
plt.close()


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print("train:", X_train.shape, "| test:", X_test.shape)


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

lin = LinearRegression()          
lin.fit(X_train, y_train)         

pred_train = lin.predict(X_train)
pred_test  = lin.predict(X_test)

def adj_r2(r2, n, p):
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

n, p = X_train.shape              
r2_train = r2_score(y_train, pred_train)

metrics = {
    "train RMSE": np.sqrt(mean_squared_error(y_train, pred_train)),
    "test RMSE":  np.sqrt(mean_squared_error(y_test, pred_test)),
    "test MAE":   mean_absolute_error(y_test, pred_test),
    "train R2":   r2_train,
    "train adj R2": adj_r2(r2_train, n, p),
    "test R2":    r2_score(y_test, pred_test),
}
print(pd.Series(metrics).round(3))


print(f"intercept = {lin.intercept_:.3f} MPa")
for name, c in zip(X.columns, lin.coef_):
    print(f"{name:13s} {c:+.4f}")


