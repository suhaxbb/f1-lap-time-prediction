# 🏎️ F1 Lap Time Predictor using FastF1 and Machine Learning

This project uses real Formula 1 qualifying session data to build a predictive model that estimates Q3 lap times based on Q1 and Q2 performances. Built using Python, FastF1, and scikit-learn, the project also simulates lap time predictions for the 2025 Japanese Grand Prix using team and driver performance factors.

---

## 📌 Project Overview

- **Goal:** Predict Q3 lap times of F1 drivers using Q1 and Q2 data.
- **Tech Stack:** Python, FastF1, pandas, numpy, scikit-learn.
- **Model:** Linear Regression.
- **Simulation:** Apply custom performance factors to simulate the 2025 Japanese GP.
- **Output:** Prints model performance, rankings, and simulated predictions to the console.

---

## 💡 Features

- Fetches real F1 qualifying session data using FastF1.
- Converts session times into numerical format (seconds).
- Cleans and preprocesses the dataset.
- Trains a Linear Regression model to predict Q3 lap times.
- Simulates future GP results using scaling factors per team and driver.
- Outputs model performance metrics (MAE, R²) and predictions.

---

## 🧪 Steps Performed

1. **Data Collection**
   - Live F1 qualifying data from 2024 and 2025 races using FastF1 API.

2. **Data Cleaning**
   - Converted Q1, Q2, Q3 times to float values (in seconds).
   - Removed incomplete or invalid entries.

3. **Feature Engineering**
   - Independent variables: Q1_sec, Q2_sec  
   - Target variable: Q3_sec

4. **Model Training**
   - Used `LinearRegression` from scikit-learn.
   - Evaluated with Mean Absolute Error and R² Score.

5. **Simulation**
   - Applied predefined performance multipliers to teams and drivers.
   - Predicted Q3 lap times for all 20 drivers in the 2025 Japanese GP.
  

