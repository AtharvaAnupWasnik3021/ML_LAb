import pandas as pd
import statistics
import matplotlib.pyplot as plt

# Step 1: Load the data
df = pd.read_excel("Lab Session Data.xlsx", sheet_name="IRCTC Stock Price")

# Convert date column to datetime (assume it's in column A or named 'Date')
df['Date'] = pd.to_datetime(df['Date'])

# Extract Day of Week
df['Day'] = df['Date'].dt.day_name()

# Step 2: Mean and Variance of column D (assume it's named 'Price')
price_data = df.iloc[:, 3]  # Column D (0-indexed)
mean_price = statistics.mean(price_data)
variance_price = statistics.variance(price_data)

print(f"Mean of price data (population mean): {mean_price:.2f}")
print(f"Variance of price data: {variance_price:.2f}")

# Step 3: Sample mean for Wednesdays
wednesday_prices = df[df['Day'] == 'Wednesday'].iloc[:, 3]
mean_wed = statistics.mean(wednesday_prices)
print(f"Sample mean for Wednesdays: {mean_wed:.2f}")
print(f"Difference from population mean: {mean_wed - mean_price:.2f}")

# Step 4: Sample mean for April
april_data = df[df['Date'].dt.month == 4].iloc[:, 3]
mean_april = statistics.mean(april_data)
print(f"Sample mean for April: {mean_april:.2f}")
print(f"Difference from population mean: {mean_april - mean_price:.2f}")

# Step 5: Probability of loss (Chg% < 0), column I assumed as Chg%
chg_col = df.iloc[:, 8]  # Column I (0-indexed)
num_loss_days = sum(chg_col.apply(lambda x: x < 0))
prob_loss = num_loss_days / len(chg_col)
print(f"Probability of making a loss: {prob_loss:.2f}")

# Step 6: Probability of profit on Wednesday (Chg% > 0 and day == Wednesday)
wed_profit_days = df[(df['Day'] == 'Wednesday') & (chg_col > 0)].shape[0]
total_wed_days = df[df['Day'] == 'Wednesday'].shape[0]
prob_profit_wed = wed_profit_days / total_wed_days
print(f"Probability of making a profit on Wednesday: {prob_profit_wed:.2f}")

# Step 7: Conditional Probability (Profit | Wednesday)
# This is same as above because we’re already conditioning on Wednesday
print(f"Conditional Probability (Profit | Wednesday): {prob_profit_wed:.2f}")

# Step 8: Scatter plot of Chg% vs Day of Week
plt.figure(figsize=(10, 6))
plt.scatter(df['Day'], chg_col, color='blue', alpha=0.6)
plt.title("Change % vs Day of the Week")
plt.xlabel("Day of Week")
plt.ylabel("Chg%")
plt.grid(True)
plt.show()
