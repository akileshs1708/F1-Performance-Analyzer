import pandas as pd
import numpy as np


class Preprocessor:
    def __init__(self, datasets):
        self.datasets = datasets
    
    def replace_null_markers(self):
        for name, df in self.datasets.items():
            df.replace(r"\\N", np.nan, inplace=True)
            df.replace("\\N", np.nan, inplace=True)
        return self.datasets
    
    def get_missing_values_report(self, df):
        missing = df.isnull().sum()
        percent_missing = (missing / len(df)) * 100
        missing_df = pd.DataFrame({
            "Missing Values": missing,
            "Percentage": percent_missing
        })
        return missing_df[missing_df["Missing Values"] > 0].sort_values(
            by="Percentage", ascending=False
        )
    
    def generate_all_missing_reports(self):
        reports = {}
        for name, df in self.datasets.items():
            reports[name] = self.get_missing_values_report(df)
        return reports
    
    def fill_numerical_missing(self):
        for name, df in self.datasets.items():
            for col in df.select_dtypes(include=["float64", "int64"]).columns:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
        return self.datasets
    
    def fill_categorical_missing(self):
        for name, df in self.datasets.items():
            for col in df.select_dtypes(include=["object"]).columns:
                if not df[col].mode().empty:
                    mode_val = df[col].mode()[0]
                    df[col] = df[col].fillna(mode_val)
        return self.datasets
    
    def convert_date_columns(self):
        if "races" in self.datasets:
            self.datasets["races"]["date"] = pd.to_datetime(
                self.datasets["races"]["date"], errors="coerce"
            )
        if "drivers" in self.datasets:
            self.datasets["drivers"]["dob"] = pd.to_datetime(
                self.datasets["drivers"]["dob"], errors="coerce"
            )
        return self.datasets
    
    def remove_duplicates(self):
        duplicate_report = {}
        for name, df in self.datasets.items():
            before = len(df)
            df.drop_duplicates(inplace=True)
            after = len(df)
            duplicate_report[name] = before - after
        return duplicate_report
    
    def preprocess_all(self):
        print("Replacing null markers...")
        self.replace_null_markers()
        
        print("Filling numerical missing values...")
        self.fill_numerical_missing()
        
        print("Filling categorical missing values...")
        self.fill_categorical_missing()
        
        print("Converting date columns...")
        self.convert_date_columns()
        
        print("Removing duplicates...")
        duplicate_report = self.remove_duplicates()
        
        for name, count in duplicate_report.items():
            if count > 0:
                print(f"{name}: Removed {count} duplicate rows")
        
        return self.datasets
    
    def get_datasets(self):
        return self.datasets