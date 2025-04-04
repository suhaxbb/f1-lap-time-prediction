import os

# Create cache folder BEFORE importing fastf1
if not os.path.exists('cache'):
    os.makedirs('cache')

# Create visuals folder for saving plots
if not os.path.exists('visuals'):
    os.makedirs('visuals')

import fastf1
fastf1.Cache.enable_cache('cache')

# Other imports
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# ----------------------- Data Fetching and Cleaning -----------------------

def fetch_f1_data(year, round_number):
    """Fetch qualifying data using FastF1"""
    try:
        quali = fastf1.get_session(year, round_number, 'Q')
        quali.load()
        results = quali.results[['DriverNumber', 'FullName', 'TeamName', 'Q1', 'Q2', 'Q3']]
        results = results.rename(columns={'FullName': 'Driver'})
        for col in ['Q1', 'Q2', 'Q3']:
            results[col + '_sec'] = results[col].apply(lambda x: x.total_seconds() if pd.notnull(x) else None)
        print("\nQualifying Results Structure:")
        print(results.head())
        return results
    except Exception as e:
        print(f"Error fetching data: {e}")
        try:
            print("DataFrame columns available:", quali.results.columns.tolist())
        except Exception:
            pass
        return None

def convert_time_to_seconds(time_str):
    if pd.isna(time_str):
        return None
    try:
        if ':' in time_str:
            minutes, seconds = time_str.split(':')
            return float(minutes) * 60 + float(seconds)
        else:
            return float(time_str)
    except (ValueError, TypeError) as e:
        print(f"Warning: Could not convert time: {time_str}, Error: {e}")
        return None

def clean_data(df):
    print("\nBefore cleaning:")
    print(df[['Driver', 'Q1', 'Q2', 'Q3']].head())
    df['Q1_sec'] = df['Q1'].apply(convert_time_to_seconds)
    df['Q2_sec'] = df['Q2'].apply(convert_time_to_seconds)
    df['Q3_sec'] = df['Q3'].apply(convert_time_to_seconds)
    print("\nAfter cleaning:")
    print(df[['Driver', 'Q1_sec', 'Q2_sec', 'Q3_sec']].head())
    return df.dropna()  # This will drop any row with missing Q1_sec, Q2_sec, or Q3_sec

# ----------------------- Visualization Functions -----------------------

def visualize_boxplot(df):
    """Boxplot for Q1, Q2, Q3 lap times"""
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df[['Q1_sec', 'Q2_sec', 'Q3_sec']])
    plt.title('Distribution of Qualifying Lap Times (seconds)')
    plt.ylabel('Lap Time (s)')
    plt.tight_layout()
    plt.savefig('visuals/boxplot_qualifying.png')
    plt.show()

def visualize_prediction_accuracy(results_df):
    """Scatter plot: Actual vs Predicted Q3 Times"""
    plt.figure(figsize=(6, 4))
    plt.scatter(results_df['Q3_sec'], results_df['Predicted_Q3'], c='blue', label='Drivers')
    plt.plot([results_df['Q3_sec'].min(), results_df['Q3_sec'].max()],
             [results_df['Q3_sec'].min(), results_df['Q3_sec'].max()],
             'r--', label='Ideal Prediction')
    plt.xlabel('Actual Q3 Lap Time (s)')
    plt.ylabel('Predicted Q3 Lap Time (s)')
    plt.title('Predicted vs Actual Q3 Lap Times')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('visuals/predicted_vs_actual.png')
    plt.show()

def visualize_japanese_gp_predictions(results_df):
    """Bar chart for predicted Q3 times for Japanese GP 2025"""
    plt.figure(figsize=(10, 6))
    plt.barh(results_df['Driver'], results_df['Predicted_Q3'], color='darkcyan')
    plt.xlabel('Predicted Q3 Lap Time (s)')
    plt.title('2025 Japanese GP - Predicted Q3 Lap Times')
    plt.gca().invert_yaxis()  # Fastest times at the top
    plt.tight_layout()
    plt.savefig('visuals/japanese_gp_prediction.png')
    plt.show()

def visualize_driver_progression(df):
    """Line chart showing driver progression from Q1 to Q3"""
    melted = df[['Driver', 'Q1_sec', 'Q2_sec', 'Q3_sec']].melt(id_vars='Driver', 
                                                                  var_name='Session', 
                                                                  value_name='Time')
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=melted, x='Session', y='Time', hue='Driver', marker="o")
    plt.title('Driver Progression: Q1 to Q3')
    plt.xlabel('Qualifying Session')
    plt.ylabel('Lap Time (s)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('visuals/driver_progression.png')
    plt.show()

# ----------------------- Model Training and Evaluation -----------------------

def train_and_evaluate(df):
    X = df[['Q1_sec', 'Q2_sec']]
    y = df['Q3_sec']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X)
    results_df = df[['Driver', 'TeamName', 'Q1_sec', 'Q2_sec', 'Q3_sec']].copy()
    results_df['Predicted_Q3'] = predictions
    results_df['Difference'] = results_df['Predicted_Q3'] - results_df['Q3_sec']
    results_df = results_df.sort_values('Predicted_Q3')
    
    print("\nPredicted Q3 Rankings:")
    print("=" * 70)
    print(f"{'Position':<10}{'Driver':<15}{'Team':<20}{'Predicted Time':<15}{'Actual Time':<15}")
    print("-" * 70)
    for idx, row in results_df.iterrows():
        pred_time = f"{row['Predicted_Q3']:.3f}"
        actual_time = f"{row['Q3_sec']:.3f}" if not pd.isna(row['Q3_sec']) else "N/A"
        print(f"{results_df.index.get_loc(idx)+1:<10}{row['Driver']:<15}{row['TeamName']:<20}{pred_time:<15}{actual_time:<15}")
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("\nModel Performance Metrics:")
    print(f'Mean Absolute Error: {mae:.2f} seconds')
    print(f'R^2 Score: {r2:.2f}')
    
    # Visualize prediction accuracy
    visualize_prediction_accuracy(results_df)
    
    return model, results_df

# ----------------------- Data Fetching for Recent Races -----------------------

def fetch_recent_data():
    """Fetch data from recent races using FastF1"""
    all_data = []
    current_year = 2025
    for round_num in range(1, 5):  # First 4 races of 2025
        print(f"Fetching data for {current_year} round {round_num}...")
        df = fetch_f1_data(current_year, round_num)
        if df is not None:
            df['Year'] = current_year
            df['Round'] = round_num
            all_data.append(df)
    print("Fetching 2024 Japanese GP data...")
    japan_2024 = fetch_f1_data(2024, 4)
    if japan_2024 is not None:
        japan_2024['Year'] = 2024
        japan_2024['Round'] = 4
        all_data.append(japan_2024)
    return all_data

# ----------------------- Prediction for Japanese GP 2025 -----------------------

def apply_performance_factors(predictions_df):
    """Apply 2025-specific performance factors"""
    base_time = 89.5  # in seconds
    team_factors = {
        'Red Bull Racing': 0.997,
        'Ferrari': 0.998,
        'McLaren': 0.999,
        'Mercedes': 0.999,
        'Aston Martin': 1.001,
        'RB': 1.002,
        'Williams': 1.003,
        'Haas F1 Team': 1.004,
        'Kick Sauber': 1.004,
        'Alpine': 1.005,
    }
    driver_factors = {
        'Max Verstappen': 0.998,
        'Charles Leclerc': 0.999,
        'Carlos Sainz': 0.999,
        'Lando Norris': 0.999,
        'Oscar Piastri': 1.000,
        'Sergio Perez': 1.000,
        'Lewis Hamilton': 1.000,
        'George Russell': 1.000,
        'Fernando Alonso': 1.000,
        'Lance Stroll': 1.001,
        'Alex Albon': 1.001,
        'Daniel Ricciardo': 1.001,
        'Yuki Tsunoda': 1.002,
        'Valtteri Bottas': 1.002,
        'Zhou Guanyu': 1.003,
        'Kevin Magnussen': 1.003,
        'Nico Hulkenberg': 1.003,
        'Logan Sargeant': 1.004,
        'Pierre Gasly': 1.004,
        'Esteban Ocon': 1.004,
    }
    for idx, row in predictions_df.iterrows():
        team_factor = team_factors.get(row['Team'], 1.005)
        driver_factor = driver_factors.get(row['Driver'], 1.002)
        base_prediction = base_time * team_factor * driver_factor
        random_variation = np.random.uniform(-0.1, 0.1)
        predictions_df.loc[idx, 'Predicted_Q3'] = base_prediction + random_variation
    return predictions_df

def predict_japanese_gp(model, latest_data):
    """Predict Q3 times for Japanese GP 2025"""
    driver_teams = {
        'Max Verstappen': 'Red Bull Racing',
        'Sergio Perez': 'Red Bull Racing',
        'Charles Leclerc': 'Ferrari',
        'Carlos Sainz': 'Ferrari',
        'Lewis Hamilton': 'Mercedes',
        'George Russell': 'Mercedes',
        'Lando Norris': 'McLaren',
        'Oscar Piastri': 'McLaren',
        'Fernando Alonso': 'Aston Martin',
        'Lance Stroll': 'Aston Martin',
        'Daniel Ricciardo': 'RB',
        'Yuki Tsunoda': 'RB',
        'Alexander Albon': 'Williams',
        'Logan Sargeant': 'Williams',
        'Valtteri Bottas': 'Kick Sauber',
        'Zhou Guanyu': 'Kick Sauber',
        'Kevin Magnussen': 'Haas F1 Team',
        'Nico Hulkenberg': 'Haas F1 Team',
        'Pierre Gasly': 'Alpine',
        'Esteban Ocon': 'Alpine'
    }
    results_df = pd.DataFrame(list(driver_teams.items()), columns=['Driver', 'Team'])
    results_df = apply_performance_factors(results_df)
    results_df = results_df.sort_values('Predicted_Q3')
    
    print("\nJapanese GP 2025 Qualifying Predictions:")
    print("=" * 100)
    print(f"{'Position':<10}{'Driver':<20}{'Team':<25}{'Predicted Q3':<15}")
    print("-" * 100)
    for idx, row in results_df.iterrows():
        print(f"{results_df.index.get_loc(idx)+1:<10}"
              f"{row['Driver']:<20}"
              f"{row['Team']:<25}"
              f"{row['Predicted_Q3']:.3f}s")
    
    # Visualize Japanese GP predictions as a bar chart
    visualize_japanese_gp_predictions(results_df)

# ----------------------- Main Script -----------------------

if __name__ == "__main__":
    print("Initializing enhanced F1 prediction model...")
    all_data = fetch_recent_data()
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        # Drop any rows with missing Q1_sec, Q2_sec, or Q3_sec values
        valid_data = combined_df.dropna(subset=['Q1_sec', 'Q2_sec', 'Q3_sec'], how='any')
        # Visualize raw lap time distributions
        visualize_boxplot(valid_data)
        
        imputer = SimpleImputer(strategy='median')
        X = valid_data[['Q1_sec', 'Q2_sec']]
        y = valid_data['Q3_sec']
        X_clean = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        y_clean = pd.Series(imputer.fit_transform(y.values.reshape(-1, 1)).ravel())
        
        model = LinearRegression()
        model.fit(X_clean, y_clean)
        
        # Evaluate model and visualize prediction accuracy
        model, results_df = train_and_evaluate(valid_data)
        
        # Predict and visualize Japanese GP 2025 Q3 times
        predict_japanese_gp(model, valid_data)
        
        # Visualize driver progression (from Q1 to Q3)
        visualize_driver_progression(valid_data)
    else:
        print("Failed to fetch F1 data")
