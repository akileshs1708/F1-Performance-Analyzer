Formula 1 Driver Performance Prediction & Analytics
A comprehensive Formula 1 data analytics and performance prediction system built in Python.

This project performs exploratory data analysis (EDA), feature engineering, visualization, and predictive modeling using historical F1 data to analyze and predict driver and constructor performance.

Table of Contents
Overview
Project Structure
Dataset
Features
1. Data Loading
2. Data Preprocessing
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Predictive Modeling
6. Visualization
Technology Stack
Installation
Running the Project
Example Outputs
Analytical Capabilities
Configuration
Extending the Project
Overview
This project provides a full pipeline for Formula 1 data analysis:

Ingests historical F1 data from CSV files.
Cleans and preprocesses relational datasets.
Engineers performance-related features for drivers and constructors.
Builds and evaluates predictive models for race and season performance.
Generates visualizations for trends, dominance, consistency, and efficiency.
The system can be used both as a research/analytics tool and as a base for building an interactive web application (e.g., Flask or Streamlit).

Project Structure
Bash

.
├── app/
│   └── app.py                     # Application entry (Flask/Streamlit-style runner)
│
├── data/
│   ├── circuits.csv
│   ├── constructors.csv
│   ├── constructor_results.csv
│   ├── constructor_standings.csv
│   ├── drivers.csv
│   ├── driver_standings.csv
│   ├── lap_times.csv
│   ├── pit_stops.csv
│   ├── qualifying.csv
│   ├── races.csv
│   ├── results.csv
│   ├── seasons.csv
│   ├── sprint_results.csv
│   └── status.csv
│
├── src/
│   ├── data_loader.py             # Data loading utilities
│   ├── preprocessing.py           # Data cleaning & transformation
│   ├── feature_engineering.py     # Feature creation
│   ├── eda.py                     # Exploratory Data Analysis functions
│   ├── models.py                  # ML model definitions & training logic
│   ├── visualizations.py          # Plot generation
│   ├── utils.py                   # Helper utilities
│   └── __init__.py
│
├── config.py                      # Global configuration
├── main.py                        # Main pipeline runner
├── requirements.txt               # Project dependencies
├── avg_lap_times.png              # Generated visualization
├── races_per_year.png             # Generated visualization
└── README.md                      # Project documentation
Dataset
The project uses historical Formula 1 data stored as CSV files in the data/ directory. The core files include:

drivers.csv – Driver information (IDs, names, codes, etc.)
constructors.csv – Constructor (team) metadata
circuits.csv – Circuit information and locations
races.csv – Race metadata (season, round, circuit, date)
results.csv – Race results per driver and constructor
lap_times.csv – Lap-by-lap timing data
pit_stops.csv – Pit stop timings and counts
qualifying.csv – Qualifying session results
sprint_results.csv – Sprint race results (where applicable)
seasons.csv – Season-level data
constructor_results.csv – Constructor race results
constructor_standings.csv – Constructor championship standings
driver_standings.csv – Driver championship standings
status.csv – Status codes (e.g., Finished, Engine, Accident, etc.)
The relational structure of these datasets allows multi-dimensional analyses of:

Driver performance
Constructor dominance
Seasonal trends
Race strategies
Lap efficiency and consistency
Features
1. Data Loading
Structured CSV ingestion from data/ directory.
Modular loading logic in src/data_loader.py.
Clean separation between raw I/O and downstream processing logic.
2. Data Preprocessing
Implemented in src/preprocessing.py:

Missing value handling.
Data type normalization (dates, numeric fields, categorical labels).
Deduplication and sanity checks.
Merging and joining of relational tables (e.g., results + races + drivers + constructors).
3. Exploratory Data Analysis (EDA)
Implemented in src/eda.py:

Races per season.
Distribution of race results and points.
Average lap time trends.
Constructor and driver dominance over seasons.
Performance distributions and trend visualizations.
4. Feature Engineering
Implemented in src/feature_engineering.py:

Driver consistency metrics
Variance/standard deviation of results and lap times.
Rolling performance indicators over recent races.
Constructor performance indicators
Average finishing position.
Points per race, podium rate, DNF rates.
Seasonal aggregates
Season-level driver and constructor statistics.
Form metrics (recent performance vs season average).
Lap-time derived performance scores
Average lap times, best lap differentials.
Lap efficiency and race pace metrics.
5. Predictive Modeling
Implemented in src/models.py:

Performance prediction models using scikit-learn.
Driver ranking estimation models.
Race outcome probability modeling (e.g., probability of finishing in top N).
Configurable training, validation, and evaluation routines.
6. Visualization
Implemented in src/visualizations.py:

Average Lap Time Trends – Evolution of lap times across races/seasons.
Races Per Year – Growth and variation in season length.
Performance Charts – Driver vs constructor performance comparisons.
Standings Visualization – Championship standings and progression.
Additional charts for consistency, dominance, and lap efficiency.
Technology Stack
Language: Python 3.x
Core Libraries:
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
Application Layer:
Flask / Streamlit-style interface via app/app.py
All dependencies are listed in requirements.txt.

Installation
Clone the Repository
Bash

git clone <repository_url>
cd <project_folder>
Create a Virtual Environment (Recommended)
Bash

python -m venv venv

# Mac/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
Install Dependencies
Bash

pip install -r requirements.txt
Ensure the data/ directory contains the CSV files listed above.

Running the Project
Run Full Pipeline
Execute the full data pipeline (loading, preprocessing, feature engineering, modeling, and/or EDA) via:

Bash

python main.py
(You can extend main.py to accept CLI arguments for different modes, e.g., --train, --eda, etc.)

Run Application Interface
To start the application interface (e.g., web dashboard):

Bash

python app/app.py
Follow the console output for the local URL and access it in your browser.

Example Outputs
Generated visualizations (stored in the project root by default):

avg_lap_times.png – Average lap time trend analysis.
races_per_year.png – Number of races per season visualization.
Additional visualizations can be created using utilities in src/visualizations.py.

Analytical Capabilities
This project enables, among others:

Driver vs Constructor Performance Comparison
Relative contributions of drivers vs teams to success.
Seasonal Dominance Trends
Identification of dominant drivers/constructors across eras.
Consistency Scoring
Quantifying driver and team consistency over races and seasons.
Race Outcome Modeling
Predicting finishing positions or probabilities (e.g., top 3/5/10).
Championship Trend Analysis
Tracking title fights and momentum over the course of a season.
Lap Efficiency Benchmarking
Comparing drivers and constructors based on lap times, race pace, and execution.
