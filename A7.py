import pandas as pd
import numpy as np
from numpy.linalg import norm
import seaborn as sns
import matplotlib.pyplot as plt

# Load Excel data
df = pd.read_excel("Lab Session Data.xlsx", sheet_name="IRCTC Stock Price")

# Select first 20 observations
df = df.head(20)

# Binary columns only (for JC and SMC)
binary_cols = [col for col in df.columns if set(df[col].dropna().unique()).issubset({0, 1})]
df_binary = df[binary_cols].astype(int)

# Numeric data for cosine
df_numeric = df.select_dtypes(include=[np.number]).dropna().head(20)

# Initialize matrices
n = len(df)
jc_matrix = np.zeros((n, n))
smc_matrix = np.zeros((n, n))
cos_matrix = np.zeros((n, n))

# Function to calculate JC and SMC
def calculate_jc_smc(v1, v2):
    f11 = np.sum((v1 == 1) & (v2 == 1))
    f00 = np.sum((v1 == 0) & (v2 == 0))
    f10 = np.sum((v1 == 1) & (v2 == 0))
    f01 = np.sum((v1 == 0) & (v2 == 1))
    jc = f11 / (f11 + f10 + f01) if (f11 + f10 + f01) > 0 else 0
    smc = (f11 + f00) / (f11 + f10 + f01 + f00) if (f11 + f10 + f01 + f00) > 0 else 0
    return jc, smc

# Fill matrices
for i in range(n):
    for j in range(n):
        if len(binary_cols) > 0:
            jc, smc = calculate_jc_smc(df_binary.iloc[i].values, df_binary.iloc[j].values)
            jc_matrix[i, j] = jc
            smc_matrix[i, j] = smc
        # Cosine similarity
        vec1 = df_numeric.iloc[i].values
        vec2 = df_numeric.iloc[j].values
        cos_sim = np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))
        cos_matrix[i, j] = cos_sim

# Create DataFrames
jc_df = pd.DataFrame(jc_matrix, columns=[f"Obs {i}" for i in range(1, n+1)], index=[f"Obs {i}" for i in range(1, n+1)])
smc_df = pd.DataFrame(smc_matrix, columns=[f"Obs {i}" for i in range(1, n+1)], index=[f"Obs {i}" for i in range(1, n+1)])
cos_df = pd.DataFrame(cos_matrix, columns=[f"Obs {i}" for i in range(1, n+1)], index=[f"Obs {i}" for i in range(1, n+1)])

# Plot Heatmaps
plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
sns.heatmap(jc_df, annot=False, cmap='YlGnBu')
plt.title("Jaccard Coefficient Heatmap")

plt.subplot(1, 3, 2)
sns.heatmap(smc_df, annot=False, cmap='YlOrRd')
plt.title("Simple Matching Coefficient Heatmap")

plt.subplot(1, 3, 3)
sns.heatmap(cos_df, annot=False, cmap='PuBuGn')
plt.title("Cosine Similarity Heatmap")

plt.tight_layout()
plt.show()
