import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.inspection import permutation_importance

os.makedirs("data", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

df = pd.read_csv("data/concrete.csv")

print("дубликатов:", df.duplicated().sum(), "| пропусков:", df.isna().sum().sum())
df = df.drop_duplicates().reset_index(drop=True)
print("после чистки:", df.shape)

X = df.drop(columns=["strength"])
y = df["strength"]

print(df.describe())

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].scatter(X["cement"], y, alpha=0.3)
axes[0].set_xlabel("cement")
axes[0].set_ylabel("strength")
axes[1].scatter(X["age"], y, alpha=0.3)
axes[1].set_xlabel("age")
axes[1].set_ylabel("strength")
plt.tight_layout()
plt.savefig("figures/A2_scatter.png")
plt.close()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("train:", X_train.shape, "| test:", X_test.shape)

lr = LinearRegression()
lr.fit(X_train, y_train)

y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
n, p = X_train.shape
adj_r2 = 1 - (1 - train_r2) * (n - 1) / (n - p - 1)

print(pd.Series({
    "train RMSE": np.sqrt(mean_squared_error(y_train, y_train_pred)),
    "test RMSE": np.sqrt(mean_squared_error(y_test, y_test_pred)),
    "test MAE": mean_absolute_error(y_test, y_test_pred),
    "train R2": train_r2,
    "train adj R2": adj_r2,
    "test R2": test_r2,
}).round(3))

print(f"intercept = {lr.intercept_:.3f} MPa")
for col, c in zip(X.columns, lr.coef_):
    print(f"{col:13s} {c:+.4f}")

residuals = y_test - y_test_pred
fig, ax = plt.subplots(figsize=(7, 4))
ax.scatter(X_test["age"], residuals, alpha=0.4)
ax.axhline(0, color="red")
ax.set_xscale("log")
ax.set_xlabel("age (log scale)")
ax.set_ylabel("residual")
plt.tight_layout()
plt.savefig("figures/A4_residuals.png")
plt.close()

poly_results = []
for deg in [1, 2, 3, 4]:
    pipe = make_pipeline(PolynomialFeatures(degree=deg, include_bias=False),
                         StandardScaler(), LinearRegression())
    pipe.fit(X_train, y_train)
    p_feat = pipe[0].n_output_features_
    ytr = pipe.predict(X_train)
    yte = pipe.predict(X_test)
    tr_r2 = r2_score(y_train, ytr)
    n_tr = len(y_train)
    adj = 1 - (1 - tr_r2) * (n_tr - 1) / (n_tr - p_feat - 1)
    poly_results.append([deg, p_feat,
                         np.sqrt(mean_squared_error(y_train, ytr)),
                         np.sqrt(mean_squared_error(y_test, yte)),
                         tr_r2, adj,
                         r2_score(y_test, yte)])

poly_df = pd.DataFrame(poly_results,
    columns=["degree","features","train_rmse","test_rmse","train_r2","adj_r2","test_r2"])
print(poly_df)

y_class = (df["strength"] >= 40).astype(int)
print(y_class.value_counts(normalize=True))

Xc_train, Xc_test, c_train, c_test = train_test_split(
    X, y_class, test_size=0.2, random_state=42, stratify=y_class)

K_values = [1, 3, 5, 7, 15, 25, 51, 101]
knn_results = []
for K in K_values:
    knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=K))
    knn.fit(Xc_train, c_train)
    knn_results.append([K, knn.score(Xc_train, c_train), knn.score(Xc_test, c_test)])

knn_df = pd.DataFrame(knn_results, columns=["K","train_acc","test_acc"])
print(knn_df)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(knn_df["K"], knn_df["train_acc"], label="train")
ax.plot(knn_df["K"], knn_df["test_acc"], label="test")
ax.set_xscale("log")
ax.set_xlabel("K (log scale)")
ax.set_ylabel("accuracy")
ax.legend()
plt.tight_layout()
plt.savefig("figures/A6_knn_sweep.png")
plt.close()

BEST_K = 7
knn_best = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=BEST_K))
knn_best.fit(Xc_train, c_train)

spread = []
for rs in [0, 1, 2, 3, 4]:
    Xtr, Xte, ctr, cte = train_test_split(X, y_class, test_size=0.2, random_state=rs, stratify=y_class)
    m = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=BEST_K))
    m.fit(Xtr, ctr)
    spread.append(m.score(Xte, cte))
print("KNN spread:", spread)

depth_results = []
for d in [1, 2, 3, 4, 5, 6, 8, None]:
    tree = DecisionTreeClassifier(criterion="entropy", max_depth=d, random_state=0)
    tree.fit(Xc_train, c_train)
    depth_results.append([d, tree.get_n_leaves(),
                          tree.score(Xc_train, c_train),
                          tree.score(Xc_test, c_test)])

depth_df = pd.DataFrame(depth_results, columns=["max_depth","n_leaves","train_acc","test_acc"])
print(depth_df)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot([str(d) for d in depth_df["max_depth"]], depth_df["train_acc"], label="train")
ax.plot([str(d) for d in depth_df["max_depth"]], depth_df["test_acc"], label="test")
ax.set_xlabel("max_depth")
ax.set_ylabel("accuracy")
ax.legend()
plt.tight_layout()
plt.savefig("figures/A6_tree_sweep.png")
plt.close()

BEST_DEPTH = 4
tree_best = DecisionTreeClassifier(criterion="entropy", max_depth=BEST_DEPTH, random_state=0)
tree_best.fit(Xc_train, c_train)
print(export_text(tree_best, feature_names=list(X.columns), max_depth=2))

def clf_metrics(name, model, Xte, cte):
    pred = model.predict(Xte)
    tn, fp, fn, tp = confusion_matrix(cte, pred).ravel()
    return {"model": name, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "accuracy": accuracy_score(cte, pred),
            "precision": precision_score(cte, pred),
            "recall": recall_score(cte, pred),
            "f1": f1_score(cte, pred)}

metrics_df = pd.DataFrame([
    clf_metrics("KNN", knn_best, Xc_test, c_test),
    clf_metrics("Tree", tree_best, Xc_test, c_test),
])
print(metrics_df)

proba = tree_best.predict_proba(Xc_test)[:, 1]
thresh_rows = []
for t in [0.3, 0.5, 0.7, 0.9]:
    pred_t = (proba >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(c_test, pred_t).ravel()
    thresh_rows.append({"threshold": t, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                        "precision": precision_score(c_test, pred_t, zero_division=0),
                        "recall": recall_score(c_test, pred_t, zero_division=0)})
print(pd.DataFrame(thresh_rows))

metrics_df.to_csv("results/A_metrics.csv", index=False)

pipe_std = make_pipeline(StandardScaler(), LinearRegression())
pipe_std.fit(X_train, y_train)
std_coef = pd.Series(pipe_std[-1].coef_, index=X.columns).sort_values(key=abs, ascending=False)
print("Standardised coefs:\n", std_coef)
print("Tree importances:\n", pd.Series(tree_best.feature_importances_, index=X.columns).sort_values(ascending=False))

perm = permutation_importance(knn_best, Xc_test, c_test, scoring="f1", n_repeats=20, random_state=0)
perm_series = pd.Series(perm.importances_mean, index=X.columns).sort_values(ascending=False)
print("Permutation importances:\n", perm_series)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
std_coef.plot(kind="bar", ax=axes[0], title="Standardised coef")
pd.Series(tree_best.feature_importances_, index=X.columns).sort_values(ascending=False).plot(kind="bar", ax=axes[1], title="Tree importance")
perm_series.plot(kind="bar", ax=axes[2], title="Permutation importance (KNN)")
plt.tight_layout()
plt.savefig("figures/A8_importances.png")
plt.close()
