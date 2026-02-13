import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import DataLoader
from src.preprocessing import Preprocessor
from src.eda import EDAAnalyzer
from src.visualizations import Visualizer
from src.feature_engineering import FeatureEngineer
from src.models import F1ModelTrainer
from src.utils import Utils

st.set_page_config(
    page_title="F1 Data Analysis Dashboard",
    page_icon="racing_car",
    layout="wide"
)


@st.cache_data
def load_and_preprocess_data():
    loader = DataLoader()
    datasets = loader.load_all_data()
    preprocessor = Preprocessor(datasets)
    datasets = preprocessor.preprocess_all()
    return datasets


@st.cache_resource
def train_models(datasets):
    feature_engineer = FeatureEngineer(datasets)
    X, y = feature_engineer.get_model_features()
    
    if X is not None and y is not None:
        trainer = F1ModelTrainer()
        trainer.train_all_models(X, y)
        return trainer, X, y
    return None, None, None


def main():
    st.title("F1 Data Analysis Dashboard")
    st.markdown("---")
    
    with st.spinner("Loading data..."):
        datasets = load_and_preprocess_data()
    
    eda = EDAAnalyzer(datasets)
    visualizer = Visualizer()
    
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select Page",
        [
            "Overview",
            "Race Statistics",
            "Driver Analysis",
            "Constructor Analysis",
            "Qualifying Analysis",
            "Pit Stop Analysis",
            "Model Predictions",
            "Championship Analysis"
        ]
    )
    
    if page == "Overview":
        render_overview_page(datasets, eda)
    elif page == "Race Statistics":
        render_race_statistics_page(eda, visualizer)
    elif page == "Driver Analysis":
        render_driver_analysis_page(datasets, eda, visualizer)
    elif page == "Constructor Analysis":
        render_constructor_analysis_page(datasets, eda, visualizer)
    elif page == "Qualifying Analysis":
        render_qualifying_analysis_page(eda, visualizer)
    elif page == "Pit Stop Analysis":
        render_pit_stop_analysis_page(eda, visualizer)
    elif page == "Model Predictions":
        render_model_predictions_page(datasets, visualizer)
    elif page == "Championship Analysis":
        render_championship_analysis_page(eda, visualizer)


def render_overview_page(datasets, eda):
    st.header("Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Races", len(datasets.get("races", [])))
    with col2:
        st.metric("Total Drivers", len(datasets.get("drivers", [])))
    with col3:
        st.metric("Total Constructors", len(datasets.get("constructors", [])))
    with col4:
        st.metric("Total Circuits", len(datasets.get("circuits", [])))
    
    st.subheader("Dataset Information")
    
    selected_dataset = st.selectbox(
        "Select Dataset",
        list(datasets.keys())
    )
    
    if selected_dataset:
        df = datasets[selected_dataset]
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"Shape: {df.shape}")
            st.write(f"Columns: {list(df.columns)}")
        
        with col2:
            st.write("Data Types:")
            st.write(df.dtypes)
        
        st.subheader("Sample Data")
        st.dataframe(df.head(10))
        
        st.subheader("Statistical Summary")
        st.dataframe(df.describe())


def render_race_statistics_page(eda, visualizer):
    st.header("Race Statistics")
    
    races_per_year = eda.get_races_per_year()
    if races_per_year is not None:
        st.subheader("Races Per Year")
        fig = visualizer.plot_races_per_year(races_per_year)
        st.pyplot(fig)
    
    avg_lap_times = eda.get_average_lap_times()
    if avg_lap_times is not None:
        st.subheader("Average Lap Times Trend")
        fig = visualizer.plot_average_lap_times(avg_lap_times)
        st.pyplot(fig)


def render_driver_analysis_page(datasets, eda, visualizer):
    st.header("Driver Analysis")
    
    top_drivers = eda.get_top_drivers(20)
    if top_drivers is not None:
        st.subheader("Top 20 Drivers by Points")
        
        drivers_df = datasets.get("drivers")
        if drivers_df is not None:
            top_drivers_with_names = top_drivers.reset_index()
            top_drivers_with_names = top_drivers_with_names.merge(
                drivers_df[["driverId", "forename", "surname"]],
                on="driverId"
            )
            top_drivers_with_names["Driver"] = (
                top_drivers_with_names["forename"] + " " +
                top_drivers_with_names["surname"]
            )
            st.dataframe(
                top_drivers_with_names[["Driver", "points", "wins"]].head(20)
            )
    
    driver_wins = eda.get_driver_wins()
    if driver_wins is not None:
        st.subheader("Driver Wins Over Time")
        fig = visualizer.plot_driver_performance(driver_wins, top_n=5)
        st.pyplot(fig)
    
    driver_consistency = eda.get_driver_consistency()
    if driver_consistency is not None:
        st.subheader("Driver Consistency")
        fig = visualizer.plot_driver_consistency(driver_consistency)
        st.pyplot(fig)


def render_constructor_analysis_page(datasets, eda, visualizer):
    st.header("Constructor Analysis")
    
    top_constructors = eda.get_top_constructors(20)
    if top_constructors is not None:
        st.subheader("Top 20 Constructors by Points")
        
        constructors_df = datasets.get("constructors")
        if constructors_df is not None:
            top_const_with_names = top_constructors.reset_index()
            top_const_with_names = top_const_with_names.merge(
                constructors_df[["constructorId", "name"]],
                on="constructorId"
            )
            st.dataframe(
                top_const_with_names[["name", "points", "wins"]].head(20)
            )
    
    constructor_wins = eda.get_constructor_wins()
    if constructor_wins is not None:
        st.subheader("Constructor Dominance Over Time")
        fig = visualizer.plot_constructor_dominance(constructor_wins, top_n=5)
        st.pyplot(fig)


def render_qualifying_analysis_page(eda, visualizer):
    st.header("Qualifying Analysis")
    
    qualifying_results = eda.get_qualifying_impact()
    if qualifying_results is not None:
        st.subheader("Impact of Starting Position on Final Result")
        fig = visualizer.plot_qualifying_impact(qualifying_results)
        st.pyplot(fig)
        
        st.subheader("Position Change Distribution")
        fig = visualizer.plot_position_change_distribution(qualifying_results)
        st.pyplot(fig)
    
    position_changes = eda.get_position_changes()
    if position_changes is not None:
        st.subheader("Drivers Who Gain Most Positions")
        st.dataframe(position_changes.head(10))


def render_pit_stop_analysis_page(eda, visualizer):
    st.header("Pit Stop Analysis")
    
    pit_performance = eda.get_pit_stop_analysis()
    if pit_performance is not None:
        st.subheader("Impact of Pit Stop Frequency on Race Position")
        fig = visualizer.plot_pit_stop_impact(pit_performance)
        st.pyplot(fig)
        
        st.subheader("Pit Stop Efficiency vs Race Position")
        fig = visualizer.plot_pit_efficiency(pit_performance)
        st.pyplot(fig)
        
        st.subheader("Pit Stop Correlation Analysis")
        corr_cols = ["stop", "milliseconds", "avg_pit_time", "positionOrder"]
        corr_data = pit_performance[corr_cols].dropna()
        fig = visualizer.plot_correlation_heatmap(
            corr_data,
            "Pit Stop Metrics Correlation"
        )
        st.pyplot(fig)


def render_model_predictions_page(datasets, visualizer):
    st.header("Model Predictions")
    
    with st.spinner("Training models..."):
        trainer, X, y = train_models(datasets)
    
    if trainer is not None:
        st.subheader("Model Comparison")
        comparison = trainer.get_model_comparison()
        st.dataframe(comparison)
        
        st.subheader("Best Model")
        st.success(f"Best performing model: {trainer.best_model_name}")
        
        best_results = trainer.results[trainer.best_model_name]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("MAE", f"{best_results['mae']:.4f}")
        with col2:
            st.metric("RMSE", f"{best_results['rmse']:.4f}")
        with col3:
            st.metric("R2 Score", f"{best_results['r2']:.4f}")
        
        feature_importance = trainer.get_feature_importance()
        if feature_importance is not None:
            st.subheader("Feature Importance")
            fig = visualizer.plot_feature_importance(
                list(X.columns),
                feature_importance
            )
            st.pyplot(fig)
        
        st.subheader("Actual vs Predicted")
        y_pred = best_results["y_pred"]
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        fig = visualizer.plot_actual_vs_predicted(y_test, y_pred)
        st.pyplot(fig)
        
        st.subheader("Feature Correlation")
        fig = visualizer.plot_correlation_heatmap(X)
        st.pyplot(fig)
    else:
        st.error("Unable to train models. Please check the data.")


def render_championship_analysis_page(eda, visualizer):
    st.header("Championship Analysis")
    
    championship_data = eda.get_championship_retention()
    if championship_data is not None:
        st.subheader("Championship Retention")
        st.metric(
            "Retention Probability",
            f"{championship_data['retention_probability']:.2f}%"
        )
        
        champions = championship_data["champions"]
        st.subheader("Championship History")
        display_cols = ["year", "surname", "points", "retained_title"]
        available_cols = [col for col in display_cols if col in champions.columns]
        st.dataframe(champions[available_cols].tail(20))
    
    champion_ages = eda.get_champion_ages()
    if champion_ages is not None:
        st.subheader("Champion Age Distribution")
        fig = visualizer.plot_champion_ages_distribution(champion_ages)
        st.pyplot(fig)
        
        st.subheader("Champion Age Trends by Decade")
        fig = visualizer.plot_champion_age_trends(champion_ages)
        st.pyplot(fig)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Average Champion Age",
                f"{champion_ages['champion_age'].mean():.1f} years"
            )
        with col2:
            st.metric(
                "Youngest Champion Age",
                f"{champion_ages['champion_age'].min()} years"
            )


if __name__ == "__main__":
    main()