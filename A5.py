import pandas as pd

# Load the Excel file and the relevant sheet
df = pd.read_excel("Lab Session Data.xlsx", sheet_name="IRCTC Stock Price")

# Step 1: Select the first 2 rows (observations)
obs1 = df.iloc[0]
obs2 = df.iloc[1]

# Step 2: Keep only binary attributes (0 or 1) — ignore continuous or non-binary features
binary_cols = [col for col in df.columns if set(df[col].dropna().unique()).issubset({0, 1})]
print("Binary columns:", binary_cols)

# Extract binary vectors
v1 = df.loc[0, binary_cols].values
v2 = df.loc[1, binary_cols].values

# Step 3: Compute f11, f00, f10, f01
f11 = sum((v1 == 1) & (v2 == 1))
f00 = sum((v1 == 0) & (v2 == 0))
f10 = sum((v1 == 1) & (v2 == 0))
f01 = sum((v1 == 0) & (v2 == 1))

# Step 4: Calculate JC and SMC
jc = f11 / (f11 + f10 + f01) if (f11 + f10 + f01) > 0 else 0
smc = (f11 + f00) / (f11 + f10 + f01 + f00) if (f11 + f10 + f01 + f00) > 0 else 0

# Step 5: Output
print(f"f11 = {f11}, f10 = {f10}, f01 = {f01}, f00 = {f00}")
print(f"Jaccard Coefficient (JC): {jc:.2f}")
print(f"Simple Matching Coefficient (SMC): {smc:.2f}")

# Step 6: Judge appropriateness
if jc > smc:
    print("JC emphasizes only matching 1's — better for sparse binary features like tags.")
else:
    print("SMC includes matching 0's — useful when 0 has meaning (e.g., absence matters).")
