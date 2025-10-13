import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv("daily_weather.csv")
data['Date']=pd.to_datetime(data['Date'])


print("First 5 rows of data:")
print(data.head())


print("\nSummary statistics:")
print(data.describe())


plt.figure(figsize=(10,5))
plt.plot(data["Date"], data["Temperature"], marker='o')
plt.xticks(rotation=45)
plt.title("Daily Temperature Trend")
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.show()

plt.figure(figsize=(8,5))
sns.histplot(data["Humidity"], bins=10, kde=True)
plt.title("Humidity Distribution")
plt.xlabel("Humidity (%)")
plt.show()

plt.figure(figsize=(8,5))
sns.scatterplot(x="Temperature", y="Humidity", data=data)
plt.title("Temperature vs Humidity")
plt.show()

plt.figure(figsize=(6,4))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show(block=True)
