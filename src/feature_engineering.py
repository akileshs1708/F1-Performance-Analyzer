import pandas as pd
import numpy as np


class FeatureEngineer:
    def __init__(self, datasets):
        self.datasets = datasets
        self.engineered_data = None
    
    def calculate_driver_consistency(self, results):
        driver_consistency = results.groupby("driverId")["positionOrder"].mean().reset_index()
        driver_consistency.rename(
            columns={"positionOrder": "avg_finish_position"},
            inplace=True
        )
        return driver_consistency
    
    def calculate_qualifying_consistency(self, results):
        qualifying_consistency = results.groupby("driverId")["grid"].mean().reset_index()
        qualifying_consistency.rename(
            columns={"grid": "avg_qualifying_position"},
            inplace=True
        )
        return qualifying_consistency
    
    def calculate_constructor_rolling_points(self, results):
        results_copy = results.copy()
        results_copy["constructor_rolling_points"] = results_copy.groupby(
            "constructorId"
        )["points"].transform(lambda x: x.rolling(5, min_periods=1).mean())
        return results_copy
    
    def calculate_dnf_rate(self, results):
        dnf_counts = results[results["statusId"] != 1].groupby("constructorId").size()
        total_counts = results.groupby("constructorId").size()
        dnf_rates = (dnf_counts / total_counts).reset_index()
        dnf_rates.columns = ["constructorId", "dnf_rate"]
        dnf_rates["dnf_rate"] = dnf_rates["dnf_rate"].fillna(0)
        return dnf_rates
    
    def calculate_overtake_difficulty(self, results):
        results_copy = results.copy()
        results_copy["overtake_difficulty"] = results_copy.groupby("raceId")["grid"].transform(
            lambda x: (x - results_copy.loc[x.index, "positionOrder"]).abs().mean()
        )
        return results_copy
    
    def engineer_all_features(self):
        results = self.datasets.get("results")
        if results is None:
            return None
        
        results_copy = results.copy()
        
        driver_consistency = self.calculate_driver_consistency(results_copy)
        results_copy = results_copy.merge(driver_consistency, on="driverId", how="left")
        
        qualifying_consistency = self.calculate_qualifying_consistency(results_copy)
        results_copy = results_copy.merge(qualifying_consistency, on="driverId", how="left")
        
        results_copy = self.calculate_constructor_rolling_points(results_copy)
        
        dnf_rates = self.calculate_dnf_rate(results_copy)
        results_copy = results_copy.merge(dnf_rates, on="constructorId", how="left")
        
        results_copy = self.calculate_overtake_difficulty(results_copy)
        
        self.engineered_data = results_copy
        return results_copy
    
    def get_model_features(self):
        if self.engineered_data is None:
            self.engineer_all_features()
        
        if self.engineered_data is None:
            return None, None
        
        feature_columns = [
            "raceId", "driverId", "constructorId", "grid", "points",
            "avg_finish_position", "avg_qualifying_position",
            "constructor_rolling_points", "dnf_rate", "overtake_difficulty"
        ]
        
        available_columns = [
            col for col in feature_columns
            if col in self.engineered_data.columns
        ]
        
        outcome = self.engineered_data[available_columns + ["positionOrder"]].copy()
        outcome = outcome.dropna()
        
        X = outcome.drop(columns=["positionOrder"])
        y = outcome["positionOrder"]
        
        return X, y
    
    def get_engineered_data(self):
        return self.engineered_data