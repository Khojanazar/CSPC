import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# DATASET 1: HEART DISEASE (CLEVELAND)

df_heart = pd.read_csv("Heart_disease_cleveland_new.csv")

# Task 1
print(f"Dataset Shape:{df_heart.shape}")
print("First rows:")
print(df_heart.head())

continuous_cols=["age","chol","trestbps","thalach"]
df_continuous=df_heart[continuous_cols]
print("\nContinuous columns isolated:",continuous_cols)

# Task 2 & 3:
fig,axes=plt.subplots(2,2,figsize=(12, 10))
axes=axes.flatten()

normality_results={}
for i,col in enumerate(continuous_cols):
    data=df_continuous[col].dropna()
    
    # Plot histogram / density
    axes[i].hist(data, bins=15, edgecolor='black', alpha=0.7, density=True)
    axes[i].set_title(f"Distribution of {col}")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Density")
    
    # Statistical normality check (Shapiro-Wilk test)
    shapiro_stat, p_value = stats.shapiro(data)
    is_normal = p_value > 0.05
    normality_results[col] = {"p_value": p_value, "normal": is_normal}
    print(f"Normality test for {col}: p-value = {p_value:.5f} -> Normally distributed? {is_normal}")

plt.tight_layout()
plt.savefig("heart_distributions.png")
plt.close()

# Task 4: Compare thalach between heart disease groups (target)

disease_thalach = df_heart[df_heart["target"] == 1]["thalach"].dropna()
healthy_thalach = df_heart[df_heart["target"] == 0]["thalach"].dropna()

# Compute means and standard errors (uncertainty)
mean_dis, sem_dis = disease_thalach.mean(), stats.sem(disease_thalach)
mean_hlth, sem_hlth = healthy_thalach.mean(), stats.sem(healthy_thalach)

print(f"Disease Group Thalach Mean:{mean_dis:.2f} ± {sem_dis:.2f} (SE)")
print(f"Healthy Group Thalach Mean:{mean_hlth:.2f} ± {sem_hlth:.2f} (SE)")

# Choose test based on normality (Mann-Whitney U if non-normal, Student's t-test if normal)
if normality_results["thalach"]["normal"]:
    t_stat,p_val=stats.ttest_ind(disease_thalach,healthy_thalach)
    print(f"Independent t-test result: t={t_stat:.4f},p={p_val:.5e}")
else:
    u_stat,p_val=stats.mannwhitneyu(disease_thalach,healthy_thalach)
    print(f"Mann-Whitney U test result: U={u_stat:.4f},p={p_val:.5e}")

# Plot means with uncertainty (error bars)
plt.figure(figsize=(6,6))
groups=['Disease (1)','Healthy (0)']
means=[mean_dis,mean_hlth]
sems=[sem_dis,sem_hlth]

plt.bar(groups,means,yerr=sems,capsize=5,color=['coral','skyblue'],alpha=0.8)
plt.ylabel("Maximum Heart Rate (thalach)")
plt.title("Max Heart Rate by Heart Disease Status")
plt.savefig("thalach_comparison.png")
plt.close()

# Task 5: Age and Thalach relationship

r_val,p_corr=stats.pearsonr(df_heart["age"],df_heart["thalach"])
print(f"Pearson correlation between age and thalach: r = {r_val:.4f}, p = {p_corr:.5e}")

plt.figure(figsize=(8, 6))
plt.scatter(df_heart["age"], df_heart["thalach"], alpha=0.6, color='purple')
plt.xlabel("Age")
plt.ylabel("Max Heart Rate (thalach)")
plt.title("Age vs. Maximum Heart Rate")
plt.savefig("age_vs_thalach.png")
plt.close()

# Bonus Task: Category unpredictability (entropy / proportions)

target_props = df_heart["target"].value_counts(normalize=True)
print("Target proportions:")
print(target_props)


# ==========================================
# DATASET 2: CHEMICAL-EXPOSURE MYSTERY
# ==========================================


df_chem = pd.read_csv("chemicals_cancer.csv")

# Task 6: Naive Analysis (Correlations with malignancy)

r_benz_naive, _ = stats.pearsonr(df_chem["benzene"], df_chem["malignancy"])
r_cad_naive, _ = stats.pearsonr(df_chem["cadmium"], df_chem["malignancy"])

print(f"Naive Correlation (Benzene vs Malignancy): {r_benz_naive:.4f}")
print(f"Naive Correlation (Cadmium vs Malignancy): {r_cad_naive:.4f}")

# Task 7: Accounting for a third variable (Confounder: pollution_index)
# Restrict data to a narrow band of pollution_index to neutralize its confounding effect
filtered_df = df_chem[(df_chem["pollution_index"] > 40) & (df_chem["pollution_index"] < 60)]

r_benz_adj, _ =stats.pearsonr(filtered_df["benzene"],filtered_df["malignancy"])
r_cad_adj, _ =stats.pearsonr(filtered_df["cadmium"],filtered_df["malignancy"])

print(f"Adjusted Correlation (Benzene vs Malignancy):{r_benz_adj:.4f}")
print(f"Adjusted Correlation (Cadmium vs Malignancy):{r_cad_adj:.4f}")