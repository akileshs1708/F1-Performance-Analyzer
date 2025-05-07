# F1-Performance-Analyzer
This project focuses on predicting Formula 1 race outcomes using historical data from 1950 to 2024. By applying machine learning techniques, feature engineering, and statistical analysis, we aim to uncover patterns and insights that contribute to performance on the track.

---

## 1. Data Exploration & Preprocessing

- **Dataset Scope:** Race data spanning 1950 to 2024, including driver performance, constructor data, circuit attributes, and weather conditions.
- **Steps Taken:**
  - Analyzed data structure and identified trends across eras.
  - Cleaned missing values, normalized inconsistent entries, and ensured temporal integrity.
  - Performed statistical tests (e.g., correlation, ANOVA, t-tests) to validate data assumptions and relevance.

---

## 2. Feature Engineering

Designed and extracted new features to enhance model prediction quality:

- **Driver Consistency**
  - Average qualifying and finishing positions
  - Historical race performance trend (sliding window)
  
- **Team Strength**
  - Constructor points by season
  - Reliability metrics (DNFs, technical issues)

- **Track Complexity**
  - Overtaking difficulty (based on historical position change stats)
  - Circuit-specific weather volatility and race incident frequency

---

## 3. Model Development & Evaluation

Built, trained, and validated machine learning models to predict finishing positions.

- **Models Used:**
  - Linear Regression
  - Random Forest
  - XGBoost / LightGBM
  - (Optional) ARIMA baseline for comparison
  
- **Evaluation Metrics:**
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
  - R² Score
  - Time-based Cross-Validation

---

## 4. Insights & Visualization

- **Key Influencers Identified:**
  - Constructor reliability and team dynamics
  - Driver adaptability to circuits and weather conditions
  - Pit stop strategies and tire management

- **Visualizations:**
  - Heatmaps: Correlation of features with race outcomes
  - Historical dominance charts: Constructors & Drivers
  - Scatterplots: Budget vs. Performance
  - Time-series plots: Evolution of team and driver performance

---

and Some problem statments are also Solved ...
