
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------
df = pd.read_csv("yourfile.csv")   # Change filename here

# Ensure datetime column is parsed correctly
df["Date"] = pd.to_datetime(df["Date"])
df["Day"] = df["Date"].dt.day_name()
df["Year"] = df["Date"].dt.year
df["Hour"] = df["Date"].dt.hour

# ----------------------------------------------------------
# 1. FILTER WEEKDAYS + PLOT PEDESTRIAN COUNTS
# ----------------------------------------------------------

weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
df_weekdays = df[df["Day"].isin(weekdays)]

weekday_counts = (
    df_weekdays.groupby("Day")["Brooklyn Bridge"]
    .sum()
    .reindex(weekdays)
)

plt.figure(figsize=(10, 5))
plt.plot(weekday_counts.index, weekday_counts.values, marker="o")
plt.title("Pedestrian Counts by Weekday (Brooklyn Bridge)")
plt.xlabel("Weekday")
plt.ylabel("Pedestrian Count")
plt.grid(True)
plt.savefig("weekday_counts_plot.png")
plt.close()

# ----------------------------------------------------------
# 2. 2019 DATA + WEATHER CORRELATION ANALYSIS
# ----------------------------------------------------------

df_2019 = df[df["Year"] == 2019].copy()

# Average pedestrians by weather summary
weather_effect = (
    df_2019.groupby("Weather Summary")["Brooklyn Bridge"]
    .mean()
    .sort_values()
)

weather_effect.to_csv("weather_effect_2019.csv")

# Correlation matrix (numeric columns only)
numeric_cols = df_2019.select_dtypes(include=np.number)
corr_matrix = numeric_cols.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix for Weather vs Pedestrian Counts (2019)")
plt.savefig("correlation_matrix_2019.png")
plt.close()

# ----------------------------------------------------------
# 3. TIME OF DAY CATEGORIZATION + PATTERN ANALYSIS
# ----------------------------------------------------------

def categorize_time_of_day(hour):
    if 5 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 21:
        return "Evening"
    else:
        return "Night"

df["TimeOfDay"] = df["Hour"].apply(categorize_time_of_day)

tod_counts = (
    df.groupby("TimeOfDay")["Brooklyn Bridge"]
    .mean()
    .reindex(["Morning", "Afternoon", "Evening", "Night"])
)

plt.figure(figsize=(8, 5))
sns.barplot(x=tod_counts.index, y=tod_counts.values)
plt.title("Pedestrian Activity by Time of Day (Brooklyn Bridge)")
plt.xlabel("Time of Day")
plt.ylabel("Average Pedestrian Count")
plt.savefig("time_of_day_activity.png")
plt.close()

tod_counts.to_csv("time_of_day_counts.csv")

print("Analysis complete. Files generated:")
print("- weekday_counts_plot.png")
print("- weather_effect_2019.csv")
print("- correlation_matrix_2019.png")
print("- time_of_day_activity.png")
print("- time_of_day_counts.csv")
