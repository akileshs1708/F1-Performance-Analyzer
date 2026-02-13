# Formula 1 Driver Performance Prediction & Analytics

A comprehensive Formula 1 data analytics and performance prediction system built with Python. This project performs exploratory data analysis (EDA), feature engineering, visualization, and predictive modeling using historical F1 data spanning multiple decades.

## 📊 Project Overview

This system leverages historical Formula 1 data to analyze driver performance, constructor dominance, race strategies, and predict future outcomes. The modular architecture allows for easy extension and integration of new analytical capabilities.

### Key Capabilities
- **Performance Analysis**: Compare drivers and constructors across different eras
- **Trend Identification**: Analyze seasonal patterns and rule-change impacts
- **Predictive Modeling**: Forecast race outcomes and driver rankings
- **Visual Analytics**: Generate insightful visualizations for presentation-ready analysis

## 🗂️ Project Structure
```text
├── app/
│ └── app.py # Web application interface (Flask/Streamlit)
│
├── data/
│ ├── circuits.csv # Circuit information
│ ├── constructors.csv # Constructor/team details
│ ├── constructor_results.csv # Constructor race results
│ ├── constructor_standings.csv # Constructor championship standings
│ ├── drivers.csv # Driver information
│ ├── driver_standings.csv # Driver championship standings
│ ├── lap_times.csv # Individual lap time data
│ ├── pit_stops.csv # Pit stop records
│ ├── qualifying.csv # Qualifying session results
│ ├── races.csv # Race calendar and information
│ ├── results.csv # Complete race results
│ ├── seasons.csv # Season information
│ ├── sprint_results.csv # Sprint race results
│ └── status.csv # Race status codes
│
├── src/
│ ├── data_loader.py # CSV ingestion and data loading
│ ├── preprocessing.py # Data cleaning and transformation
│ ├── feature_engineering.py # Feature creation and selection
│ ├── eda.py # Exploratory data analysis
│ ├── models.py # ML model implementations
│ ├── visualizations.py # Plot generation utilities
│ ├── utils.py # Helper functions
│ └── init.py
│
├── config.py # Configuration parameters
├── main.py # Main pipeline execution
├── requirements.txt # Project dependencies
├── avg_lap_times.png # Generated visualization
├── races_per_year.png # Generated visualization
└── README.md # Project documentation
```
