import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import pandas as pd


class Visualizer:
    def __init__(self):
        self.fig_size_large = (12, 6)
        self.fig_size_medium = (10, 5)
        self.fig_size_small = (8, 4)
    
    def plot_races_per_year(self, races_per_year):
        fig, ax = plt.subplots(figsize=self.fig_size_large)
        ax.plot(races_per_year.index, races_per_year.values, marker="o")
        ax.set_xlabel("Year")
        ax.set_ylabel("Number of Races")
        ax.set_title("F1 Races Per Year (1950-2024)")
        ax.grid(True)
        return fig
    
    def plot_average_lap_times(self, avg_lap_times):
        fig, ax = plt.subplots(figsize=self.fig_size_large)
        ax.plot(
            avg_lap_times.index,
            avg_lap_times.values / 1000,
            marker="o",
            color="r"
        )
        ax.set_xlabel("Year")
        ax.set_ylabel("Average Lap Time (seconds)")
        ax.set_title("Trend of Average Lap Times in F1")
        ax.grid(True)
        return fig
    
    def plot_constructor_dominance(self, constructor_wins, top_n=10):
        fig, ax = plt.subplots(figsize=(14, 8))
        top_constructors = constructor_wins.sum().nlargest(top_n).index
        constructor_wins[top_constructors].plot(ax=ax)
        ax.set_xlabel("Year")
        ax.set_ylabel("Number of Wins")
        ax.set_title("Constructor Dominance in F1")
        ax.legend(title="Constructor", loc="upper left", fontsize=8)
        return fig
    
    def plot_driver_performance(self, driver_wins, top_n=10):
        fig, ax = plt.subplots(figsize=(14, 8))
        top_drivers = driver_wins.sum().nlargest(top_n).index
        driver_wins[top_drivers].plot(ax=ax)
        ax.set_xlabel("Year")
        ax.set_ylabel("Number of Wins")
        ax.set_title("Top Driver Performance in F1")
        ax.legend(title="Driver", loc="upper left", fontsize=8)
        return fig
    
    def plot_correlation_heatmap(self, data, title="Feature Correlation Heatmap"):
        fig, ax = plt.subplots(figsize=(10, 8))
        corr_matrix = data.corr()
        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            linewidths=0.5,
            ax=ax
        )
        ax.set_title(title)
        return fig
    
    def plot_qualifying_impact(self, qualifying_results):
        fig, ax = plt.subplots(figsize=self.fig_size_medium)
        sns.scatterplot(
            data=qualifying_results,
            x="grid",
            y="positionOrder",
            alpha=0.5,
            ax=ax
        )
        ax.set_xlabel("Starting Grid Position")
        ax.set_ylabel("Final Race Position")
        ax.set_title("Impact of Starting Position on Final Race Result")
        return fig
    
    def plot_position_change_distribution(self, qualifying_results):
        fig, ax = plt.subplots(figsize=self.fig_size_medium)
        sns.histplot(qualifying_results["position_change"], bins=20, kde=True, ax=ax)
        ax.set_xlabel("Positions Gained or Lost")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Position Changes from Qualifying to Race")
        return fig
    
    def plot_pit_stop_impact(self, pit_performance):
        fig, ax = plt.subplots(figsize=self.fig_size_medium)
        sns.boxplot(data=pit_performance, x="stop", y="positionOrder", ax=ax)
        ax.set_xlabel("Number of Pit Stops")
        ax.set_ylabel("Final Race Position")
        ax.set_title("Impact of Pit Stop Frequency on Race Position")
        ax.invert_yaxis()
        return fig
    
    def plot_pit_efficiency(self, pit_performance):
        fig, ax = plt.subplots(figsize=self.fig_size_medium)
        sns.scatterplot(
            data=pit_performance,
            x="avg_pit_time",
            y="positionOrder",
            alpha=0.5,
            ax=ax
        )
        ax.set_xlabel("Average Pit Stop Time (ms)")
        ax.set_ylabel("Final Race Position")
        ax.set_title("Impact of Pit Stop Efficiency on Race Position")
        ax.invert_yaxis()
        return fig
    
    def plot_driver_consistency(self, driver_performance):
        fig, ax = plt.subplots(figsize=self.fig_size_medium)
        driver_performance.columns = ["driverId", "avg_position", "std_position"]
        sns.scatterplot(
            data=driver_performance,
            x="avg_position",
            y="std_position",
            ax=ax
        )
        ax.set_xlabel("Average Race Position")
        ax.set_ylabel("Standard Deviation of Position")
        ax.set_title("Driver Consistency in Race Performance")
        return fig
    
    def plot_team_performance(self, team_performance):
        fig, ax = plt.subplots(figsize=self.fig_size_large)
        top_teams = team_performance.nlargest(15, "points")
        sns.barplot(data=top_teams, x="constructorId", y="points", ax=ax)
        ax.set_xlabel("Constructor ID")
        ax.set_ylabel("Total Points")
        ax.set_title("Team Performance Comparison")
        plt.xticks(rotation=45)
        return fig
    
    def plot_driver_transitions(self, merged_data):
        driver_movements = merged_data[["forename", "surname", "constructorId", "name"]]
        transitions = []
        
        for driver in driver_movements.groupby(["forename", "surname"]):
            driver_name = f"{driver[0][0]} {driver[0][1]}"
            teams = driver[1]["name"].unique()
            for i in range(len(teams) - 1):
                transitions.append((driver_name, teams[i], teams[i + 1]))
        
        G = nx.DiGraph()
        for transition in transitions:
            G.add_edge(transition[1], transition[2], label=transition[0])
        
        fig, ax = plt.subplots(figsize=(14, 10))
        pos = nx.spring_layout(G, k=2, iterations=50)
        nx.draw(
            G, pos,
            with_labels=True,
            node_size=2000,
            node_color="lightblue",
            font_size=8,
            font_weight="bold",
            arrows=True,
            ax=ax
        )
        ax.set_title("Driver Transitions Across Teams")
        return fig
    
    def plot_champion_ages_distribution(self, champion_data):
        fig, ax = plt.subplots(figsize=self.fig_size_medium)
        sns.histplot(
            champion_data["champion_age"],
            bins=10,
            kde=True,
            color="blue",
            alpha=0.7,
            ax=ax
        )
        ax.axvline(
            champion_data["champion_age"].mean(),
            color="red",
            linestyle="--",
            label="Mean Age"
        )
        ax.set_title("Distribution of Championship-Winning Ages")
        ax.set_xlabel("Age at Championship Win")
        ax.set_ylabel("Frequency")
        ax.legend()
        return fig
    
    def plot_champion_age_trends(self, champion_data):
        fig, ax = plt.subplots(figsize=self.fig_size_medium)
        age_trends = champion_data.groupby("decade")["champion_age"].mean()
        sns.lineplot(
            x=age_trends.index.astype(str),
            y=age_trends.values,
            marker="o",
            linestyle="-",
            color="green",
            ax=ax
        )
        ax.set_title("Average Age of F1 Champions Over Decades")
        ax.set_xlabel("Decade")
        ax.set_ylabel("Average Age at Championship Win")
        plt.xticks(rotation=45)
        return fig
    
    def plot_feature_importance(self, feature_names, importances):
        fig, ax = plt.subplots(figsize=self.fig_size_medium)
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        }).sort_values("importance", ascending=True)
        
        ax.barh(importance_df["feature"], importance_df["importance"])
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")
        ax.set_title("Feature Importance")
        return fig
    
    def plot_actual_vs_predicted(self, y_test, y_pred):
        fig, ax = plt.subplots(figsize=self.fig_size_medium)
        ax.scatter(y_test, y_pred, alpha=0.5)
        ax.plot(
            [y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()],
            "r--",
            lw=2
        )
        ax.set_xlabel("Actual Position")
        ax.set_ylabel("Predicted Position")
        ax.set_title("Actual vs Predicted Race Positions")
        return fig