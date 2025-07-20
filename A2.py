import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Step 1: Load data
df = pd.read_excel("Lab Session Data.xlsx", sheet_name="Purchase data") # Corrected sheet name

# Step 2: Create 'Class' column based on 'Payment (Rs)'
df['Class'] = df['Payment (Rs)'].apply(lambda x: 'RICH' if x > 200 else 'POOR')

# Drop rows with missing values that would affect feature or label creation
df_cleaned = df[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)', 'Class']].dropna()


# Step 3: Split data
X = df_cleaned[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']]  # Features (product quantities)
y = df_cleaned['Class']      # Labels

# Encode labels: RICH=1, POOR=0
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train classifier
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Step 5: Predict and Evaluate
y_pred = clf.predict(X_test)

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=le.classes_))
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
