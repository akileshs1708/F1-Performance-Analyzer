import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



circuits = pd.read_csv("circuits.csv")
constructors = pd.read_csv("constructors.csv")
constructor_results = pd.read_csv("constructor_results.csv")
constructor_standings = pd.read_csv("constructor_standings.csv")
driver_standings = pd.read_csv("driver_standings.csv")
drivers = pd.read_csv("drivers.csv")
lap_times = pd.read_csv("lap_times.csv")
pit_stops = pd.read_csv("pit_stops.csv")
qualifying = pd.read_csv("qualifying.csv")
races = pd.read_csv("races.csv")
results = pd.read_csv("results.csv")
seasons = pd.read_csv("seasons.csv")
sprint_results = pd.read_csv("sprint_results.csv")
status = pd.read_csv("status.csv")

datasets = {
    "circuits": circuits,
    "constructors": constructors,
    "constructor_results": constructor_results,
    "constructor_standings": constructor_standings,
    "driver_standings": driver_standings,
    "drivers": drivers,
    "lap_times": lap_times,
    "pit_stops": pit_stops,
    "qualifying": qualifying,
    "races": races,
    "results": results,
    "seasons": seasons,
    "sprint_results": sprint_results,
    "status": status
}



# Replace '\N' with NaN for easier handling
for name, df in datasets.items():
    df.replace(r"\\N", np.nan, inplace=True)

# Function to print missing values percentage
def missing_values_report(df):
    missing = df.isnull().sum()
    percent_missing = (missing / len(df)) * 100
    missing_df = pd.DataFrame({"Missing Values": missing, "Percentage": percent_missing})
    return missing_df[missing_df["Missing Values"] > 0].sort_values(by="Percentage", ascending=False)


# Display missing values report for all datasets
for name, df in datasets.items():
    print(f"\n--- {name.upper()} ---")
    print(missing_values_report(df))

# Handling missing values
# 1. Fill missing numerical values with median (if applicable)
for name, df in datasets.items():
    for col in df.select_dtypes(include=["float64", "int64"]).columns:
        df.fillna({col: df[col].median()}, inplace=True)

# 2. Fill categorical missing values with mode (most frequent value)
for name, df in datasets.items():
    for col in df.select_dtypes(include=["object"]).columns:
        df.fillna({col: df[col].mode()[0]}, inplace=True)

# Convert data types
races["date"] = pd.to_datetime(races["date"])
drivers["dob"] = pd.to_datetime(drivers["dob"])

# Remove duplicates if any
for name, df in datasets.items():
    before = len(df)
    df.drop_duplicates(inplace=True)
    after = len(df)
    print(f"{name}: Removed {before - after} duplicate rows.")

# Recheck missing values after cleaning
for name, df in datasets.items():
    print(f"\n--- {name.upper()} --- After Cleaning")
    print(missing_values_report(df))

for name, df in datasets.items():
    print(f"\n{name.upper()} Dataset")
    print(df.info())
    print(df.describe())
    print("\n")

# Race per Year
races_per_year = races.groupby("year").size()
plt.figure(figsize=(12, 6))
plt.plot(races_per_year.index, races_per_year.values,marker='o')
plt.xlabel("Year")
plt.ylabel("Number of Races")
plt.title("F1 Races Per Year (1950-2024)")
plt.grid(True)
plt.show()

# Average lap times
lap_times = pd.read_csv("lap_times.csv")
lap_times["milliseconds"] = pd.to_numeric(lap_times["milliseconds"], errors="coerce")
# Merge with race year
lap_times = lap_times.merge(races[["raceId", "year"]], on="raceId")
# Compute average lap time per year
avg_lap_time = lap_times.groupby("year")["milliseconds"].mean()
plt.figure(figsize=(12, 6))
plt.plot(avg_lap_time.index, avg_lap_time.values / 1000,marker='o', color="r")
plt.xlabel("Year")
plt.ylabel("Average Lap Time (seconds)")
plt.title("Trend of Average Lap Times in F1 (1950-2024)")
plt.grid(True)
plt.show()

# Wins per constructor per year
constructor_wins = results[results["positionOrder"] == 1].merge(races[["raceId", "year"]], on="raceId")
constructor_wins = constructor_wins.merge(constructors[["constructorId", "name"]], on="constructorId")
constructor_wins = constructor_wins.groupby(["year", "name"]).size().unstack(fill_value=0)
# Plot
constructor_wins.plot()
plt.xlabel("Year")
plt.ylabel("Number of Wins")
plt.title("Constructor Dominance in F1 (1950-2024)")
plt.legend(title="Constructor", loc="upper left")
plt.show()

# Drivers Performance
driver_wins = results[results["positionOrder"] == 1].merge(races[["raceId", "year"]], on="raceId")
driver_wins = driver_wins.merge(drivers[["driverId", "surname"]], on="driverId")
driver_wins = driver_wins.groupby(["year", "surname"]).size().unstack(fill_value=0)
# Plot
driver_wins.plot()
plt.xlabel("Year")
plt.ylabel("Number of Wins")
plt.title("Driver Performance in F1 (1950-2024)")
plt.legend(title="Driver", loc="upper left")
plt.show()


 # Calculate average finishing position per driver
driver_consistency = results.groupby("driverId")["positionOrder"].mean().reset_index()
driver_consistency.rename(columns={"positionOrder": "avg_finish_position"}, inplace=True)
results = results.merge(driver_consistency, on="driverId", how="left")

# Average qualifying position
qualifying_consistency = results.groupby("driverId")["grid"].mean().reset_index()
qualifying_consistency.rename(columns={"grid": "avg_qualifying_position"}, inplace=True)
results = results.merge(qualifying_consistency, on="driverId", how="left")

# Compute rolling average of constructor points over the last 5 races
results["constructor_rolling_points"] = results.groupby("constructorId")["points"].transform(lambda x: x.rolling(5, min_periods=1).mean())

# Count DNF (Did Not Finish) instances per constructor
dnf_rates = results[results["statusId"] != 1].groupby("constructorId").size() / results.groupby("constructorId").size()
dnf_rates = dnf_rates.reset_index().rename(columns={0: "dnf_rate"})
results = results.merge(dnf_rates, on="constructorId", how="left")

results["overtake_difficulty"] = results.groupby("raceId")["grid"].transform(lambda x: (x - results["positionOrder"]).abs().mean())

print(list(results.columns))

outcome = results[['raceId', 'driverId', 'constructorId','grid','positionOrder', 'points', 'avg_finish_position', 'avg_qualifying_position', 'constructor_rolling_points', 'dnf_rate', 'overtake_difficulty']]
print(outcome)


# Features and target variable
X = outcome.drop(columns=["positionOrder"])
y = outcome["positionOrder"]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
y_pred = rf_model.predict(X_test)

# Evaluation Metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R² Score: {r2}")

# Cross-Validation
cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring="neg_mean_absolute_error")
print(f"Cross-Validation MAE: {-cv_scores.mean()}")


# Compute correlation matrix
corr_matrix = outcome.corr()

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

#def driverPerformances(counstructor_staandings,driver_performance):
# Analyze constructors' performance
constructor_performance = constructor_standings.groupby('constructorId').agg({'points': 'sum', 'wins': 'sum'}).sort_values(by='points', ascending=False)
print("Top Constructors by Points:")
print(constructor_performance.head(10))

# Analyze drivers' performance
driver_performance = driver_standings.groupby('driverId').agg({'points': 'sum', 'wins': 'sum'}).sort_values(by='points', ascending=False)
print("Top Drivers by Points:")
print(driver_performance.head(10))

# Relationship between races and wins
sns.scatterplot(data=driver_performance, x=driver_performance.index, y='wins', alpha=0.7)
plt.xlabel("Driver ID")
plt.ylabel("Total Wins")
plt.title("Driver Performance: Races vs Wins")
plt.xticks(rotation=90)
plt.show()

sns.scatterplot(data=constructor_performance, x=constructor_performance.index, y='wins', alpha=0.7)
plt.xlabel("Constructor ID")
plt.ylabel("Total Wins")
plt.title("Constructor Performance: Races vs Wins")
plt.xticks(rotation=90)
plt.show()


#def qualifyingFactor(counstructor_staandings,driver_performance):
# Qualifying vs. Race Performance
qualifying_results = qualifying.merge(results, on=['raceId', 'driverId'])
qualifying_results['position_change'] = qualifying_results['grid'] - qualifying_results['positionOrder']

# Average positions gained or lost
position_changes = qualifying_results.groupby('driverId').agg({'position_change': 'mean'}).sort_values(by='position_change', ascending=False)
print("Drivers who gain the most positions on average:")
print(position_changes.head(10))

# Visualizing impact of starting position on final result
sns.scatterplot(data=qualifying_results, x='grid', y='positionOrder', alpha=0.5)
plt.xlabel("Starting Grid Position")
plt.ylabel("Final Race Position")
plt.title("Impact of Starting Position on Final Race Result")
plt.show()

# Distribution of position changes
sns.histplot(qualifying_results['position_change'], bins=20, kde=True)
plt.xlabel("Positions Gained or Lost")
plt.ylabel("Frequency")
plt.title("Distribution of Position Changes from Qualifying to Race")
plt.show()

#def pitStop(counstructor_staandings,driver_performance):
# Pit Stop Strategies: Evaluating Optimal Pit Stop Frequency and Timing for Race Success
pit_stop_analysis = pit_stops.groupby(['raceId', 'driverId']).agg({'stop': 'count', 'milliseconds': 'sum'}).reset_index()
race_results = results[['raceId', 'driverId', 'positionOrder']]
pit_performance = pit_stop_analysis.merge(race_results, on=['raceId', 'driverId'])

plt.figure(figsize=(10, 5))
sns.boxplot(data=pit_performance, x='stop', y='positionOrder')
plt.xlabel("Number of Pit Stops")
plt.ylabel("Final Race Position")
plt.title("Impact of Pit Stop Frequency on Race Position")
plt.gca().invert_yaxis()
plt.show()

# Analyzing Pit Stop Efficiency and Its Influence on Race Outcomes
pit_performance['avg_pit_time'] = pit_performance['milliseconds'] / pit_performance['stop']

plt.figure(figsize=(10, 5))
sns.scatterplot(data=pit_performance, x='avg_pit_time', y='positionOrder', alpha=0.5)
plt.xlabel("Average Pit Stop Time (ms)")
plt.ylabel("Final Race Position")
plt.title("Impact of Pit Stop Efficiency on Race Position")
plt.gca().invert_yaxis()
plt.show()

correlation = pit_performance[['stop', 'milliseconds', 'avg_pit_time', 'positionOrder']].corr()
print("Correlation Matrix:")
print(correlation)
    
#def headToHead(counstructor_staandings,driver_performance):
# Head-to-Head Driver Analysis: Identifying Competitive Rivalries
head_to_head = results.groupby(['raceId', 'driverId'])[['positionOrder']].min().reset_index()
head_to_head['wins'] = head_to_head.groupby('driverId')['positionOrder'].transform(lambda x: (x == x.min()).astype(int))
head_to_head_summary = head_to_head.groupby('driverId').agg({'wins': 'sum', 'positionOrder': 'count'}).rename(columns={'positionOrder': 'races'}).sort_values(by='wins', ascending=False)
print("Top Head-to-Head Drivers:")
print(head_to_head_summary.head(10))
    
#def hypotheticalDriverSwaps():
# Hypothetical Driver Swaps: Predicting Impact on Standings
def swap_drivers(driver1, driver2):
    swapped_results = results.copy()
    swapped_results.loc[swapped_results['driverId'] == driver1, 'driverId'] = -1
    swapped_results.loc[swapped_results['driverId'] == driver2, 'driverId'] = driver1
    swapped_results.loc[swapped_results['driverId'] == -1, 'driverId'] = driver2
    updated_standings = swapped_results.groupby('driverId').agg({'points': 'sum'}).sort_values(by='points', ascending=False)
    return updated_standings

print("Standings after hypothetical swap:")
print(swap_drivers(1, 2).head(10))  # Replace 1 and 2 with actual driver IDs


# Driver Movements & Team Networks: Mapping Driver Transitions
merged = results.merge(drivers, on='driverId').merge(constructors, on='constructorId')
driver_movements = merged[['forename', 'surname', 'constructorId', 'name']]
transitions = []
for driver in driver_movements.groupby(['forename', 'surname']):
    driver_name = f"{driver[0][0]} {driver[0][1]}"
    teams = driver[1]['name'].unique()
    for i in range(len(teams) - 1):
        transitions.append((driver_name, teams[i], teams[i + 1]))
G = nx.DiGraph()
for transition in transitions:
    G.add_edge(transition[1], transition[2], label=transition[0])
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G)  # positions for all nodes
nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue', font_size=10, font_weight='bold', arrows=True)
edge_labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
plt.title('Driver Transitions Across Teams')
plt.show()

# Team Performance Comparison: Success Rates vs. Different Opponents
team_performance = constructor_standings.groupby(['constructorId']).agg({'points': 'sum', 'wins': 'sum'}).reset_index()
sns.barplot(data=team_performance, x='constructorId', y='points')
plt.xlabel("Constructor ID")
plt.ylabel("Total Points")
plt.title("Team Performance Comparison")
plt.show()

# Driver Consistency in Race Performance
driver_performance = results.groupby('driverId').agg({'positionOrder': ['mean', 'std']}).reset_index()
driver_performance.columns = ['driverId', 'avg_position', 'std_position']
sns.scatterplot(data=driver_performance, x='avg_position', y='std_position')
plt.xlabel("Average Race Position")
plt.ylabel("Standard Deviation of Position")
plt.title("Driver Consistency in Race Performance")
plt.show()

# Lap Time Efficiency
lap_efficiency = lap_times.groupby(['raceId', 'driverId']).agg({'milliseconds': 'mean'}).reset_index()
sns.boxplot(data=lap_efficiency, x='driverId', y='milliseconds')
plt.xlabel("Constructor ID")
plt.ylabel("Average Lap Time (ms)")
plt.title("Lap Time Efficiency Across Circuits")
plt.show()

# Best Team Lineup: Selecting Drivers Based on Performance Trends
best_drivers = driver_standings.groupby('driverId').agg({'points': 'sum'}).sort_values(by='points', ascending=False).head(2)
print("Best possible team lineup:")
print(best_drivers)

# Predictions for 2025 Season
def predict_winner():
    predicted_winner = driver_standings.groupby('driverId').agg({'points': 'sum'}).sort_values(by='points', ascending=False).head(1)
    return predicted_winner

print("Predicted Driver Champion for 2025:")
print(predict_winner())

# Struggling Teams Analysis
struggling_teams = constructor_standings.groupby('constructorId').agg({'points': 'sum'}).sort_values(by='points').head(3)
print("Teams likely to struggle in 2025:")
print(struggling_teams)

# Driver-Specific Track Struggles
driver_circuit_performance = results.groupby(['driverId', 'raceId']).agg({'positionOrder': 'mean'}).reset_index()
sns.boxplot(data=driver_circuit_performance, x='driverId', y='positionOrder')
plt.xlabel("Driver ID")
plt.ylabel("Average Race Position")
plt.title("Driver Performance Across Circuits")
plt.show()


# Championship Retention Probability
df = driver_standings.merge(races[["raceId", "year"]], on="raceId")
season_champions = df.groupby("year").apply(lambda x: x.loc[x["points"].idxmax()]).reset_index(drop=True)

season_champions = season_champions.merge(drivers[["driverId", "surname"]], on="driverId")
season_champions["next_year_champ"] = season_champions["surname"].shift(-1)
season_champions["retained_title"] = season_champions["surname"] == season_champions["next_year_champ"]

# Championship retention rate
total_seasons = len(season_champions) - 1  # Exclude last season
retained_count = season_champions["retained_title"].sum()
retention_probability = retained_count / total_seasons * 100
print(f"Championship Retention Probability (Overall): {retention_probability:.2f}%")
season_champions["decade"] = (season_champions["year"] // 10) * 10
decade_stats = season_champions.groupby("decade")["retained_title"].mean() * 100
print("\nDecade-wise Retention Probability:")
print(decade_stats)
multi_champions = season_champions[season_champions["retained_title"]].groupby("surname").size().sort_values(ascending=False)
print("\nMost Back-to-Back Championships:")
print(multi_champions.head(10))
df = driver_standings.merge(races[["raceId", "year"]], on="raceId")
# Identify champion for each season (driver with most points)
season_champions = df.groupby("year").apply(lambda x: x.loc[x["points"].idxmax()]).reset_index(drop=True)
# Merge with driver birthdates
season_champions = season_champions.merge(drivers[["driverId", "surname", "dob"]], on="driverId")
# Convert date of birth to datetime and calculate age at championship win
season_champions["dob"] = pd.to_datetime(season_champions["dob"])
season_champions["champion_age"] = season_champions["year"] - season_champions["dob"].dt.year
# Group by decade
season_champions["decade"] = (season_champions["year"] // 10) * 10
age_trends = season_champions.groupby("decade")["champion_age"].mean()
plt.figure(figsize=(10, 5))

# Histogram of ages at championship wins
sns.histplot(season_champions["champion_age"], bins=10, kde=True, color="blue", alpha=0.7)
plt.axvline(season_champions["champion_age"].mean(), color="red", linestyle="--", label="Mean Age")
plt.title("Distribution of Championship-Winning Ages")
plt.xlabel("Age at Championship Win")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# Decade-wise age trends
plt.figure(figsize=(10, 5))
sns.lineplot(x=age_trends.index.astype(str), y=age_trends.values, marker="o", linestyle="-", color="green")
plt.title("Average Age of F1 Champions Over Decades")
plt.xlabel("Decade")
plt.ylabel("Average Age at Championship Win")
plt.xticks(rotation=45)
plt.show()

# Display the data
print("\nDecade-wise Average Championship-Winning Age:")
print(age_trends)