import pandas as pd
import numpy as np
from numpy.linalg import norm

# Load the Excel file and sheet
df = pd.read_excel("Lab Session Data.xlsx", sheet_name="IRCTC Stock Price")

# Step 1: Select only numeric columns (ignore date/text)
df_numeric = df.select_dtypes(include=[np.number]).dropna()

# Step 2: Extract first two observation vectors (row 0 and row 1)
vec1 = df_numeric.iloc[0].values
vec2 = df_numeric.iloc[1].values

# Step 3: Compute cosine similarity
dot_product = np.dot(vec1, vec2)
norm_vec1 = norm(vec1)
norm_vec2 = norm(vec2)

cosine_similarity = dot_product / (norm_vec1 * norm_vec2)

# Output
print(f"Cosine Similarity between Observation 1 and 2: {cosine_similarity:.4f}")
