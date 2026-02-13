import pandas as pd
import numpy as np


class EDAAnalyzer:
    def __init__(self, datasets):
        self.datasets = datasets
    
    def get_dataset_info(self, name):
        if name in self.datasets:
            df = self.datasets[name]
            return {
                "shape": df.shape,
                "columns": list(df.columns),
                "dtypes": df.dtypes.to_dict(),
                "describe": df.describe()
            }
        return None
    
    def get_races_per_year(self):
        races = self.datasets.get("races")
        if races is not None:
            return races.groupby("year").size()
        return None
    
    def get_average_lap_times(self):
        lap_times = self.datasets.get("lap_times")
        races = self.datasets.get("races")
        
        if lap_times is not None and races is not None:
            lap_times_copy = lap_times.copy()
            lap_times_copy["milliseconds"] = pd.to_numeric(
                lap_times_copy["milliseconds"], errors="coerce"
            )
            merged = lap_times_copy.merge(races[["raceId", "year"]], on="raceId")
            return merged.groupby("year")["milliseconds"].mean()
        return None
    
    def get_constructor_wins(self):
        results = self.datasets.get("results")
        races = self.datasets.get("races")
        constructors = self.datasets.get("constructors")
        
        if all(d is not None for d in [results, races, constructors]):
            wins = results[results["positionOrder"] == 1].merge(
                races[["raceId", "year"]], on="raceId"
            )
            wins = wins.merge(
                constructors[["constructorId", "name"]], on="constructorId"
            )
            return wins.groupby(["year", "name"]).size().unstack(fill_value=0)
        return None
    
    def get_driver_wins(self):
        results = self.datasets.get("results")
        races = self.datasets.get("races")
        drivers = self.datasets.get("drivers")
        
        if all(d is not None for d in [results, races, drivers]):
            wins = results[results["positionOrder"] == 1].merge(
                races[["raceId", "year"]], on="raceId"
            )
            wins = wins.merge(drivers[["driverId", "surname"]], on="driverId")
            return wins.groupby(["year", "surname"]).size().unstack(fill_value=0)
        return None
    
    def get_top_constructors(self, n=10):
        constructor_standings = self.datasets.get("constructor_standings")
        if constructor_standings is not None:
            return constructor_standings.groupby("constructorId").agg({
                "points": "sum",
                "wins": "sum"
            }).sort_values(by="points", ascending=False).head(n)
        return None
    
    def get_top_drivers(self, n=10):
        driver_standings = self.datasets.get("driver_standings")
        if driver_standings is not None:
            return driver_standings.groupby("driverId").agg({
                "points": "sum",
                "wins": "sum"
            }).sort_values(by="points", ascending=False).head(n)
        return None
    
    def get_qualifying_impact(self):
        qualifying = self.datasets.get("qualifying")
        results = self.datasets.get("results")
        
        if qualifying is not None and results is not None:
            merged = qualifying.merge(results, on=["raceId", "driverId"])
            merged["position_change"] = merged["grid"] - merged["positionOrder"]
            return merged
        return None
    
    def get_position_changes(self):
        qualifying_results = self.get_qualifying_impact()
        if qualifying_results is not None:
            return qualifying_results.groupby("driverId").agg({
                "position_change": "mean"
            }).sort_values(by="position_change", ascending=False)
        return None
    
    def get_pit_stop_analysis(self):
        pit_stops = self.datasets.get("pit_stops")
        results = self.datasets.get("results")
        
        if pit_stops is not None and results is not None:
            pit_analysis = pit_stops.groupby(["raceId", "driverId"]).agg({
                "stop": "count",
                "milliseconds": "sum"
            }).reset_index()
            
            race_results = results[["raceId", "driverId", "positionOrder"]]
            merged = pit_analysis.merge(race_results, on=["raceId", "driverId"])
            merged["avg_pit_time"] = merged["milliseconds"] / merged["stop"]
            return merged
        return None
    
    def get_driver_consistency(self):
        results = self.datasets.get("results")
        if results is not None:
            return results.groupby("driverId").agg({
                "positionOrder": ["mean", "std"]
            }).reset_index()
        return None
    
    def get_championship_retention(self):
        driver_standings = self.datasets.get("driver_standings")
        races = self.datasets.get("races")
        drivers = self.datasets.get("drivers")
        
        if all(d is not None for d in [driver_standings, races, drivers]):
            df = driver_standings.merge(races[["raceId", "year"]], on="raceId")
            season_champions = df.groupby("year").apply(
                lambda x: x.loc[x["points"].idxmax()]
            ).reset_index(drop=True)
            
            season_champions = season_champions.merge(
                drivers[["driverId", "surname"]], on="driverId"
            )
            season_champions["next_year_champ"] = season_champions["surname"].shift(-1)
            season_champions["retained_title"] = (
                season_champions["surname"] == season_champions["next_year_champ"]
            )
            
            total_seasons = len(season_champions) - 1
            retained_count = season_champions["retained_title"].sum()
            retention_probability = retained_count / total_seasons * 100
            
            return {
                "champions": season_champions,
                "retention_probability": retention_probability
            }
        return None
    
    def get_champion_ages(self):
        driver_standings = self.datasets.get("driver_standings")
        races = self.datasets.get("races")
        drivers = self.datasets.get("drivers")
        
        if all(d is not None for d in [driver_standings, races, drivers]):
            df = driver_standings.merge(races[["raceId", "year"]], on="raceId")
            season_champions = df.groupby("year").apply(
                lambda x: x.loc[x["points"].idxmax()]
            ).reset_index(drop=True)
            
            season_champions = season_champions.merge(
                drivers[["driverId", "surname", "dob"]], on="driverId"
            )
            season_champions["dob"] = pd.to_datetime(
                season_champions["dob"], errors="coerce"
            )
            season_champions["champion_age"] = (
                season_champions["year"] - season_champions["dob"].dt.year
            )
            season_champions["decade"] = (season_champions["year"] // 10) * 10
            
            return season_champions
        return None