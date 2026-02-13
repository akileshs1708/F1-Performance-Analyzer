from src.data_loader import DataLoader
from src.preprocessing import Preprocessor
from src.eda import EDAAnalyzer
from src.visualizations import Visualizer
from src.feature_engineering import FeatureEngineer
from src.models import F1ModelTrainer
from src.utils import Utils
import matplotlib.pyplot as plt


def main():
    print("Loading data...")
    loader = DataLoader()
    datasets = loader.load_all_data()
    
    print("\nPreprocessing data...")
    preprocessor = Preprocessor(datasets)
    datasets = preprocessor.preprocess_all()
    
    print("\nGenerating missing values reports...")
    reports = preprocessor.generate_all_missing_reports()
    for name, report in reports.items():
        if not report.empty:
            print(f"\n{name.upper()}:")
            print(report)
    
    print("\nPerforming EDA...")
    eda = EDAAnalyzer(datasets)
    
    races_per_year = eda.get_races_per_year()
    avg_lap_times = eda.get_average_lap_times()
    top_constructors = eda.get_top_constructors()
    top_drivers = eda.get_top_drivers()
    
    print("\nTop 10 Constructors by Points:")
    print(top_constructors)
    
    print("\nTop 10 Drivers by Points:")
    print(top_drivers)
    
    print("\nEngineering features...")
    feature_engineer = FeatureEngineer(datasets)
    X, y = feature_engineer.get_model_features()
    
    if X is not None and y is not None:
        print(f"Feature matrix shape: {X.shape}")
        print(f"Target variable shape: {y.shape}")
        
        print("\nTraining models...")
        trainer = F1ModelTrainer()
        split_data = trainer.train_all_models(X, y)
        
        print("\nModel Comparison:")
        comparison = trainer.get_model_comparison()
        print(comparison)
        
        print(f"\nBest Model: {trainer.best_model_name}")
        
        feature_importance = trainer.get_feature_importance()
        if feature_importance is not None:
            print("\nFeature Importances:")
            for name, importance in zip(X.columns, feature_importance):
                print(f"  {name}: {importance:.4f}")
    
    print("\nGenerating visualizations...")
    visualizer = Visualizer()
    
    if races_per_year is not None:
        fig = visualizer.plot_races_per_year(races_per_year)
        plt.savefig("races_per_year.png")
        plt.close()
    
    if avg_lap_times is not None:
        fig = visualizer.plot_average_lap_times(avg_lap_times)
        plt.savefig("avg_lap_times.png")
        plt.close()
    
    print("\nAnalysis complete!")
    
    championship_data = eda.get_championship_retention()
    if championship_data is not None:
        print(f"\nChampionship Retention Probability: {championship_data['retention_probability']:.2f}%")
    
    champion_ages = eda.get_champion_ages()
    if champion_ages is not None:
        print(f"\nAverage Champion Age: {champion_ages['champion_age'].mean():.1f} years")


if __name__ == "__main__":
    main()