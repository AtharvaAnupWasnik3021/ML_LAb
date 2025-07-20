import pandas as pd
import numpy as np

# Step 1: Load the data
file_path = "Lab Session Data.xlsx"
df = pd.read_excel(file_path, sheet_name="Purchase data") # Corrected sheet name

# Display the first few rows to verify
print("Data Preview:\n", df.head())

# Step 2: Assume last column is total cost (C), rest are products purchased (A)
# Select only the relevant numerical columns for A and the cost column C
# Drop rows with any NaN values in the selected columns for the analysis
df_cleaned = df[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)', 'Payment (Rs)']].dropna()

A = df_cleaned[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']].values
C = df_cleaned['Payment (Rs)'].values   # Only the last column

# Step 3: Vector space dimensionality and number of vectors
dimensionality = A.shape[1]
num_vectors = A.shape[0]

# Step 4: Rank of Matrix A
rank_A = np.linalg.matrix_rank(A)

# Step 5: Pseudo-inverse to find cost vector X
A_pinv = np.linalg.pinv(A)
X = A_pinv.dot(C)  # Solves AX = C

# Print results
print("\n--- Analysis Results ---")
print(f"Dimensionality of vector space: {dimensionality}")
print(f"Number of vectors in the space: {num_vectors}")
print(f"Rank of matrix A: {rank_A}")
print(f"\nEstimated cost of each product:\n{X}")
