import pandas as pd
import numpy as np

# Load the data
df = pd.read_excel("Lab Session Data.xlsx", sheet_name="IRCTC Stock Price")

# Step 1: Identify columns with missing values
missing_cols = df.columns[df.isnull().any()]
print("Columns with missing values:", list(missing_cols))

# Step 2: Imputation process
for col in missing_cols:
    if df[col].dtype == 'object':  # Categorical
        mode_val = df[col].mode()[0]
        df[col].fillna(mode_val, inplace=True)
        print(f"Filled missing values in '{col}' using MODE: {mode_val}")

    elif df[col].dtype in ['float64', 'int64']:  # Numeric
        # Check for outliers using IQR
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]

        if len(outliers) > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"Filled missing values in '{col}' using MEDIAN: {median_val}")
        else:
            mean_val = df[col].mean()
            df[col].fillna(mean_val, inplace=True)
            print(f"Filled missing values in '{col}' using MEAN: {mean_val:.2f}")

# Step 3: Confirm no nulls remain
print("\nNull values remaining per column:")
print(df.isnull().sum())
