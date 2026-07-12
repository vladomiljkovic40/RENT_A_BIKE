# -*- coding: utf-8 -*-

import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def load_dataset():
    """
    Loads dataset from hour.csv.
    Returns:
        DataFrame: Loaded dataset
    """
    print(" Loading dataset...")
    try:
        df = pd.read_csv('hour.csv')
        if 'dteday' in df.columns:
            df['dteday'] = pd.to_datetime(df['dteday'])
        print(f"Dataset loaded: hour.csv")
        print(f"Dimensions: {df.shape}")
        print(f"Missing values: {df.isnull().sum().sum()}")
        return df
    except FileNotFoundError:
        print("Error: hour.csv file not found!")
        print("Please place hour.csv in the same directory as this script.")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def clean_data(df):
    """
    Cleans data by dropping unnecessary columns, handling outliers, and filling missing values.
    Args:
        df: Input DataFrame
    Returns:
        DataFrame: Cleaned dataset
    """
    print(" Cleaning data...")

    # Drop unnecessary columns
    columns_to_drop = ['instant', 'casual', 'registered']
    for col in columns_to_drop:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
            print(f"Dropped column: {col}")

    if 'cnt' not in df.columns:
        print("Error: 'cnt' column not found in dataset!")
        return None

    # Handle outliers using clipping instead of removal
    continuous_cols = ['temp', 'atemp', 'hum', 'windspeed', 'cnt']
    for col in continuous_cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            if IQR > 0:
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers_before = len(df[(df[col] < lower_bound) | (df[col] > upper_bound)])
                df[col] = df[col].clip(lower_bound, upper_bound)
                if outliers_before > 0:
                    print(f"Clipped {outliers_before} outliers in column {col}")

    # Fill missing values
    numeric_means = df.select_dtypes(include=[np.number]).mean()
    df[numeric_means.index] = df[numeric_means.index].fillna(numeric_means)

    return df

def exploratory_data_analysis(df):
    """
    Performs exploratory data analysis on the dataset.
    Args:
        df: Input DataFrame
    """
    print(" Running exploratory data analysis...")

    # Correlation analysis
    numeric_df = df.select_dtypes(include=[np.number])
    correlations = numeric_df.corr()['cnt'].abs().sort_values(ascending=False)
    print("Top 5 correlations with 'cnt':")
    for feature, corr in correlations.head(6).items():
        if feature != 'cnt':
            print(f"  {feature}: {corr:.3f}")

    # Summary statistics
    print(f"Average demand: {df['cnt'].mean():.0f} bikes")
    if 'hr' in df.columns:
        peak_hour = df.groupby('hr')['cnt'].mean().idxmax()
        print(f"Peak hour: {peak_hour}:00")

    # Seasonal analysis
    if 'season' in df.columns:
        seasonal_stats = df.groupby('season')['cnt'].mean()
        print("Average demand by season:")
        season_names = {1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'}
        for season, avg in seasonal_stats.items():
            print(f"  {season_names.get(season, season)}: {avg:.0f} bikes")

    # Visualizations
    try:
        # Correlation matrix heatmap
        plt.figure(figsize=(16, 6))
        plt.subplot(1, 2, 1)
        mask = np.triu(np.ones_like(numeric_df.corr(), dtype=bool))
        sns.heatmap(numeric_df.corr(), mask=mask, annot=True, cmap='coolwarm', center=0, 
                    fmt='.2f', annot_kws={'size': 10}, square=True, linewidths=0.5)
        plt.title('Correlation Matrix', fontsize=16, fontweight='bold')
        plt.xticks(fontsize=11, rotation=45, ha='right')
        plt.yticks(fontsize=11, rotation=0)

        # Boxplot for outlier detection
        plt.subplot(1, 2, 2)
        available_cols = [col for col in ['temp', 'hum', 'windspeed', 'cnt'] if col in df.columns]
        df[available_cols].boxplot(figsize=(8, 6), fontsize=11)
        plt.title('Boxplot for Outlier Detection', fontsize=16, fontweight='bold')
        plt.ylabel('Value', fontsize=12)
        plt.xlabel('Variables', fontsize=12)
        plt.xticks(fontsize=11, rotation=0)
        plt.yticks(fontsize=10)

        plt.tight_layout()
        plt.savefig('correlation_and_outliers.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Visualizations saved to 'correlation_and_outliers.png'")
    except Exception as e:
        print(f"Error creating visualizations: {e}")

def feature_engineering(df):
    """
    Generates new features from existing dataset variables.
    Args:
        df: Input DataFrame
    Returns:
        DataFrame: Dataset with newly engineered features
    """
    print(" Engineering features...")

    # Cyclic transformations for hour
    if 'hr' in df.columns:
        df['hr_sin'] = np.sin(2 * np.pi * df['hr'] / 24)
        df['hr_cos'] = np.cos(2 * np.pi * df['hr'] / 24)
        print("Added: hr_sin, hr_cos")

    # Cyclic transformations for month
    if 'mnth' in df.columns:
        df['mnth_sin'] = np.sin(2 * np.pi * (df['mnth'] - 1) / 12)
        df['mnth_cos'] = np.cos(2 * np.pi * (df['mnth'] - 1) / 12)
        print("Added: mnth_sin, mnth_cos")

    # Rush hour indicator
    if 'hr' in df.columns:
        df['rush_hour'] = ((df['hr'].between(7, 9)) | (df['hr'].between(17, 19))).astype(int)
        print("Added: rush_hour")

    # Temperature categorization
    if 'temp' in df.columns:
        df['temp_category'] = pd.cut(df['temp'], bins=[0, 0.3, 0.7, 1.0], labels=[0, 1, 2], include_lowest=True).astype(int)
        print("Added: temp_category")

    return df

def encode_categorical(df):
    """
    Encodes categorical features using OneHotEncoder.
    Args:
        df: Input DataFrame
    Returns:
        DataFrame: Encoded dataset
    """
    print(" Encoding categorical variables...")

    categorical_cols = ['season', 'weathersit', 'weekday']
    available_categorical = [col for col in categorical_cols if col in df.columns]

    if available_categorical:
        try:
            encoder = OneHotEncoder(drop='first', sparse_output=False)  # Returns a standard dense numpy array
            encoded_cols = encoder.fit_transform(df[available_categorical])
            encoded_df = pd.DataFrame(encoded_cols, 
                                      columns=encoder.get_feature_names_out(available_categorical),  # Maps generated feature names
                                      index=df.index)

            df = df.drop(available_categorical, axis=1) # Remove original categorical columns
            df = pd.concat([df, encoded_df], axis=1)    # Append encoded features to the main DataFrame
            print(f"Added {encoded_cols.shape[1]} encoded columns")
        except Exception as e:
            print(f"Encoding error: {e}")

    return df

def prepare_data(df):
    """
    Splits data chronologically to set up training and testing sets.
    Args:
        df: Input DataFrame
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    print(" Preparing data (temporal split)...")

    # Chronological split to prevent data leakage from future records
    # (80% train, 20% test)
    df = df.sort_values('dteday')
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()  
    test_df = df.iloc[split_idx:].copy()

    print(f"Training period: {train_df['dteday'].min()} to {train_df['dteday'].max()}")
    print(f"Test period: {test_df['dteday'].min()} to {test_df['dteday'].max()}")

    train_df = train_df.drop('dteday', axis=1)      # Remove date column since it's no longer needed
    test_df = test_df.drop('dteday', axis=1)    

    X_train = train_df.drop('cnt', axis=1)
    X_test = test_df.drop('cnt', axis=1)
    y_train = train_df['cnt']
    y_test = test_df['cnt']

    # Remove redundant base columns if cyclic transformations exist
    redundant_cols = ['hr', 'mnth'] if 'hr_sin' in X_train.columns else []
    for col in redundant_cols:
        if col in X_train.columns:
            X_train = X_train.drop(col, axis=1)
            X_test = X_test.drop(col, axis=1)
            print(f"Removed redundant '{col}' column")

    # Handle missing values
    numeric_means = X_train.select_dtypes(include=[np.number]).mean()
    X_train = X_train.fillna(numeric_means)
    X_test = X_test.fillna(numeric_means)

    # Log transformation for skewed target values
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)

    print(f"Features: {X_train.shape[1]} columns")
    print(f"Training samples: {X_train.shape[0]} rows")
    print(f"Test samples: {X_test.shape[0]} rows")

    return X_train, X_test, y_train_log, y_test_log

# Includes cross-validation setup
def train_model(X_train, y_train):
    """
    Trains and tunes multiple regression models.
    Args:
        X_train: Training features
        y_train: Training target
    Returns:
        dict: Tuned estimators and parameters
    """
    print(" Training and optimizing models...")
    trained_models = {}

    # Baseline evaluation of tree-based ensemble algorithms
    base_models = {
        'RandomForest': RandomForestRegressor(n_estimators=50, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=50, random_state=42)
    }

    print("Baseline algorithm evaluation:")

    for name, model in base_models.items():
        model.fit(X_train, y_train)
        cv_scores = cross_val_score(model, X_train, np.expm1(y_train), cv=3, scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(-cv_scores.mean())
        print(f"  {name}: CV RMSE = {cv_rmse:.1f}")

    # Grid Search optimization for Random Forest
    print("\nGrid Search optimization for Random Forest...")
    rf_params = {
        'n_estimators': [50, 100],
        'max_depth': [8, 10],
        'min_samples_split': [5, 10]
    }

    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    rf_grid = GridSearchCV(rf, rf_params, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)  
    rf_grid.fit(X_train, y_train)                                                            # Hyperparameter tuning via CV

    print(f"Best RF parameters: {rf_grid.best_params_}")

    # Grid Search optimization for Gradient Boosting
    print("Grid Search optimization for Gradient Boosting...")
    gb_params = {
        'n_estimators': [50, 100],
        'max_depth': [6, 8],
        'learning_rate': [0.1, 0.05]
    }

    gb = GradientBoostingRegressor(random_state=42)                                         
    gb_grid = GridSearchCV(gb, gb_params, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)      
    gb_grid.fit(X_train, y_train)                                                            # Hyperparameter tuning via CV

    trained_models = {
        'RandomForest_Opt': {
            'model': rf_grid.best_estimator_,
            'params': rf_grid.best_params_
        },
        'GradientBoosting_Opt': {
            'model': gb_grid.best_estimator_,
            'params': gb_grid.best_params_
        }
    }

    print(f"Best GB parameters: {rf_grid.best_params_}")

    return trained_models

def evaluate_model(models, X_train, X_test, y_train, y_test, model_name):
    """
    Evaluates model performance metrics.
    Args:
        models: Dictionary of trained estimators
        X_train, X_test: Feature sets
        y_train, y_test: Target arrays (log scale)
        model_name: Identifier for the model
    Returns:
        dict: Performance metrics
    """
    model = models[model_name]['model']

    # Generate predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Inverse transform from log scale
    y_train_orig = np.expm1(y_train)
    y_test_orig = np.expm1(y_test)
    y_pred_train_orig = np.expm1(y_pred_train)
    y_pred_test_orig = np.expm1(y_pred_test)

    # Compute evaluation metrics
    train_r2 = r2_score(y_train_orig, y_pred_train_orig)
    test_r2 = r2_score(y_test_orig, y_pred_test_orig)
    test_rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_test_orig))
    test_mae = mean_absolute_error(y_test_orig, y_pred_test_orig)

    metrics = {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'overfitting': abs(train_r2 - test_r2)
    }

    return metrics

def show_feature_importance(trained_models, feature_names):
    """
    Extracts and displays feature importance for the selected model.
    Args:
        trained_models: Dictionary of trained estimators
        feature_names: List of column names
    Returns:
        DataFrame: Importance ranking table
    """
    print("\n Analyzing feature importance...")

    # Evaluate the primary optimized model
    best_model_name = list(trained_models.keys())[1]
    model = trained_models[best_model_name]['model']

    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("Top 10 most important features:")
    for _, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.3f}")

    return feature_importance

def feature_selection_comparison(trained_models, feature_importance, X_train, X_test, y_train, y_test):
    """
    Compares model performance using different subsets of top features.
    """
    print("\n Comparing performance across different feature counts...")

    best_model_name = list(trained_models.keys())[0]
    best_params = trained_models[best_model_name]['params']

    feature_counts = [5, 10, len(X_train.columns)]
    results = {}

    for n_features in feature_counts:
        top_features = feature_importance.head(n_features)['feature'].tolist()
        X_train_sel = X_train[top_features]
        X_test_sel = X_test[top_features]
        label = f"Top {n_features}"

        # Train model using feature subsets
        model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
        model.fit(X_train_sel, y_train)
        y_pred = model.predict(X_test_sel)

        # Inverse transform from log scale
        y_test_orig = np.expm1(y_test)
        y_pred_orig = np.expm1(y_pred)
        r2 = r2_score(y_test_orig, y_pred_orig)

        results[label] = r2
        print(f"  {label} features: R² = {r2:.3f}")

    return results

def show_sample_predictions(y_test, y_pred_test, n_samples=5):
    """
    Prints comparative samples of actual vs predicted values.
    """
    print("\n Sample predictions...")

    y_test_orig = np.expm1(y_test)
    y_pred_test_orig = np.expm1(y_pred_test)

    n_samples = min(n_samples, len(y_test))
    sample_indices = np.random.choice(len(y_test), n_samples, replace=False)

    print(f"{'#':<3} {'Actual':<10} {'Predicted':<12} {'Error'}")
    print("-" * 35)

    for i, idx in enumerate(sample_indices):
        actual = int(y_test_orig.iloc[idx])
        predicted = int(y_pred_test_orig[idx])
        error = abs(actual - predicted)
        print(f"{i+1:<3} {actual:<10} {predicted:<12} {error}")

def main():
    """
    Main execution pipeline.
    """
    print("=== BIKE SHARING DEMAND PREDICTOR ===\n")

    try:
        # Load dataset
        df = load_dataset()
        if df is None:
            return None

        # Clean data
        df = clean_data(df)
        if df is None:
            return None

        # Run EDA
        exploratory_data_analysis(df.copy())

        # Feature engineering
        df = feature_engineering(df)

        # Categorical encoding
        df = encode_categorical(df)

        # Split and prepare data
        X_train, X_test, y_train, y_test = prepare_data(df)

        # Train models
        trained_models = train_model(X_train, y_train)

        # Evaluate models
        print("\nEvaluating models...")
        results = {}

        # Loop through all available models
        for name in trained_models:
            results[name] = evaluate_model(trained_models, X_train, X_test, y_train, y_test, name)
            
            # Label performance quality
            r2 = results[name]['test_r2']
            if r2 > 0.8:
                quality = "EXCELLENT"
            elif r2 > 0.7:
                quality = "GOOD"
            else:
                quality = "SATISFACTORY"

            overfitting = results[name]['overfitting']
            stability = "stable" if overfitting < 0.1 else "overfitting"
            
            print(f"{name}:")
            print(f"  Test R²: {r2:.3f} ({quality})")
            print(f"  Test RMSE: {results[name]['test_rmse']:.1f}")
            print(f"  Test MAE: {results[name]['test_mae']:.1f}")
            print(f"  Stability: {stability}")
            
        # Feature importance ranking
        feature_importance = show_feature_importance(trained_models, X_train.columns)

        # Feature selection comparison
        fs_results = feature_selection_comparison(trained_models, feature_importance, 
                                                 X_train, X_test, y_train, y_test)

        # Generate sample predictions
        best_model_name = max(results.keys(), key=lambda k: results[k]['test_r2'])
        best_model = trained_models[best_model_name]['model']
        y_pred_test = best_model.predict(X_test)
        show_sample_predictions(y_test, y_pred_test)

        best_r2 = results[best_model_name]['test_r2']
        best_rmse = results[best_model_name]['test_rmse']

        print(f"\nBest model: {best_model_name}")
        print(f"Test R²: {best_r2:.3f}")
        print(f"Test RMSE: {best_rmse:.1f} bikes")

        quality = "EXCELLENT" if best_r2 > 0.8 else "GOOD" if best_r2 > 0.7 else "SATISFACTORY"
        print(f"Performance: {quality}")

        print("=" * 60)

        return trained_models, results, feature_importance

    except Exception as e:
        print(f"\nError in main execution: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = main()
    if result is not None:
        print("\nProgram completed successfully!")
    else:
        print("\nProgram failed to execute.")
