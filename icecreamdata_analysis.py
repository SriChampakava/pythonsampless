import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# Step 1: Load your dataset
df = pd.read_csv(r"D:\IceCreamData.csv")

# Step 2: Display basic info
print("\n=== Dataset Info ===")
print(df.info())

# Step 3: Display first few rows
print("\n=== First 5 rows ===")
print(df.head())

# Step 4: Show summary statistics
print("\n=== Summary Statistics ===")
print(df.describe())

# Step 5: Correlation between Temperature and Revenue
print("\n=== Correlation ===")
print(df.corr())
import matplotlib.pyplot as plt
import seaborn as sns

# Scatter plot with regression line
sns.regplot(x="Temperature", y="Revenue", data=df)
plt.title("Temperature vs Ice Cream Sales")
plt.show()

# Histogram for Temperature
plt.figure(figsize=(6,4))
sns.histplot(df["Temperature"], bins=20, kde=True, color="orange")
plt.title("Temperature Distribution")
plt.show()

# Scatter Plot: Temperature vs. Revenue
plt.figure(figsize=(7,5))
sns.scatterplot(x="Temperature", y="Revenue", data=df)
plt.title("Temperature vs. Ice Cream Revenue")
plt.show()

corr = df.corr()
print("Correlation Matrix:\n", corr)

plt.figure(figsize=(5,4))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Interpretation
r = corr.loc["Temperature", "Revenue"]
print(f"\nCorrelation Coefficient (r) between Temperature and Revenue: {r:.2f}")
if r > 0.7:
    print(" Strong Positive Correlation — Higher temperature increases sales.")
elif r > 0.4:
    print(" Moderate Correlation — Temperature somewhat influences sales.")
else:
    print(" Weak Correlation — Temperature has little effect.")

# Prepare data
df = pd.read_csv(r"D:\IceCreamData.csv")
X = df[['Temperature']]
y = df['Revenue']

# Fit model
model = LinearRegression()
model.fit(X, y)

# Predictions
y_pred = model.predict(X)

# Model coefficients
print(f"Intercept: {model.intercept_:.2f}")
print(f"Coefficient: {model.coef_[0]:.2f}")

# R² score
r2 = r2_score(y, y_pred)
print(f"R² score: {r2:.3f}")

# Plot regression line
plt.figure(figsize=(8,5))
sns.scatterplot(x='Temperature', y='Revenue', data=df, color='lightblue')
plt.plot(df['Temperature'], y_pred, color='red', linewidth=2)
plt.title("Linear Regression: Temperature vs Revenue")
plt.show()

# Example: predict revenue for 30°C day
temp = pd.DataFrame({'Temperature': [30]})
predicted_revenue = model.predict(temp)
print(f"Predicted Revenue at 30°C: {predicted_revenue[0]:.2f}")


print("""
 Business Insights:
----------------------
1. Temperature and Revenue show a strong positive correlation — as temperature rises, ice cream sales increase.
2. The regression model explains about {:.1f}% of revenue variation (R² × 100).
3. For every 1°C rise, revenue increases by approximately {:.2f} units.
4. Ideal marketing opportunities occur during high-temperature periods.
5. Recommendations:
   • Stock more inventory during summer months.
   • Run discount offers on mild days to maintain steady sales.
   • Use weather forecasts to plan production and delivery.

""".format(r2*100, model.coef_[0]))
 

