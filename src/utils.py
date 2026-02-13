import pandas as pd
import numpy as np


class Utils:
    @staticmethod
    def swap_drivers(results, driver1, driver2):
        swapped_results = results.copy()
        swapped_results.loc[swapped_results["driverId"] == driver1, "driverId"] = -1
        swapped_results.loc[swapped_results["driverId"] == driver2, "driverId"] = driver1
        swapped_results.loc[swapped_results["driverId"] == -1, "driverId"] = driver2
        updated_standings = swapped_results.groupby("driverId").agg({
            "points": "sum"
        }).sort_values(by="points", ascending=False)
        return updated_standings
    
    @staticmethod
    def predict_champion(driver_standings):
        predicted_winner = driver_standings.groupby("driverId").agg({
            "points": "sum"
        }).sort_values(by="points", ascending=False).head(1)
        return predicted_winner
    
    @staticmethod
    def get_struggling_teams(constructor_standings, n=3):
        struggling_teams = constructor_standings.groupby("constructorId").agg({
            "points": "sum"
        }).sort_values(by="points").head(n)
        return struggling_teams
    
    @staticmethod
    def get_head_to_head(results):
        head_to_head = results.groupby(["raceId", "driverId"])[["positionOrder"]].min().reset_index()
        head_to_head["wins"] = head_to_head.groupby("driverId")["positionOrder"].transform(
            lambda x: (x == x.min()).astype(int)
        )
        head_to_head_summary = head_to_head.groupby("driverId").agg({
            "wins": "sum",
            "positionOrder": "count"
        }).rename(columns={"positionOrder": "races"}).sort_values(by="wins", ascending=False)
        return head_to_head_summary
    
    @staticmethod
    def format_time(milliseconds):
        seconds = milliseconds / 1000
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}:{remaining_seconds:.3f}"
    
    @staticmethod
    def calculate_points_gap(standings):
        standings_sorted = standings.sort_values(by="points", ascending=False)
        standings_sorted["points_gap"] = standings_sorted["points"].diff().abs()
        return standings_sorted