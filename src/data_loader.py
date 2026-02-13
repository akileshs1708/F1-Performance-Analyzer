import pandas as pd
import os
from config import DATA_DIR, DATA_FILES


class DataLoader:
    def __init__(self):
        self.datasets = {}
    
    def load_all_data(self):
        for name, filename in DATA_FILES.items():
            filepath = os.path.join(DATA_DIR, filename)
            if os.path.exists(filepath):
                self.datasets[name] = pd.read_csv(filepath)
                print(f"Loaded {name}: {len(self.datasets[name])} rows")
            else:
                print(f"Warning: {filename} not found")
        return self.datasets
    
    def load_single_dataset(self, name):
        if name in DATA_FILES:
            filepath = os.path.join(DATA_DIR, DATA_FILES[name])
            if os.path.exists(filepath):
                return pd.read_csv(filepath)
        return None
    
    def get_dataset(self, name):
        return self.datasets.get(name, None)
    
    def get_all_datasets(self):
        return self.datasets