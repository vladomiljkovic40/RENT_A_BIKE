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
    Učitava dataset iz hour.csv fajla.
    Returns:
        DataFrame: Učitani dataset
    """
    print(" Učitavanje dataseta...")
    try:
        df = pd.read_csv('hour.csv')
        if 'dteday' in df.columns:
            df['dteday'] = pd.to_datetime(df['dteday'])
        print(f"Dataset učitan: hour.csv")
        print(f"Dimenzije: {df.shape}")
        print(f"Nedostajuće vrednosti: {df.isnull().sum().sum()}")
        return df
    except FileNotFoundError:
        print("Greška: hour.csv fajl nije pronađen!")
        print("Molim stavite hour.csv fajl u isti folder kao program.")
        return None
    except Exception as e:
        print(f"Greška pri učitavanju: {e}")
        return None

def clean_data(df):
    """
    Čisti podatke - uklanja nepotrebne kolone, proverava anomalije i popunjava missing values.
    Args:
        df: DataFrame sa podacima
    Returns:
        DataFrame: Očišćeni dataset
    """
    print(" Čišćenje podataka...")

    # Ukloni nepotrebne kolone
    columns_to_drop = ['instant', 'casual', 'registered']
    for col in columns_to_drop:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
            print(f"Uklonjena kolona: {col}")

    if 'cnt' not in df.columns:
        print("Greška: Kolona 'cnt' nije pronađena u datasetu!")
        return None

    # Obrada anomalija (clip umesto uklanjanja)
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
                    print(f"Clip-ovano {outliers_before} outliers u koloni {col}")

    # Popuni missing values
    numeric_means = df.select_dtypes(include=[np.number]).mean()
    df[numeric_means.index] = df[numeric_means.index].fillna(numeric_means)

    return df

def exploratory_data_analysis(df):
    """
    Izvodi eksplorativnu analizu skupa podataka.
    Args:
        df: DataFrame sa podacima
    """
    print(" Eksplorativna analiza...")

    # Korelaciona analiza
    numeric_df = df.select_dtypes(include=[np.number])
    correlations = numeric_df.corr()['cnt'].abs().sort_values(ascending=False)
    print("Top 5 korelacija sa 'cnt':")
    for feature, corr in correlations.head(6).items():
        if feature != 'cnt':
            print(f"  {feature}: {corr:.3f}")

    # Osnovna statistika
    print(f"Prosečna potražnja: {df['cnt'].mean():.0f} bicikala")
    if 'hr' in df.columns:
        peak_hour = df.groupby('hr')['cnt'].mean().idxmax()
        print(f"Peak hour: {peak_hour}:00")

    # Sezonska analiza
    if 'season' in df.columns:
        seasonal_stats = df.groupby('season')['cnt'].mean()
        print("Prosečna potražnja po sezonama:")
        season_names = {1: 'Proleće', 2: 'Leto', 3: 'Jesen', 4: 'Zima'}
        for season, avg in seasonal_stats.items():
            print(f"  {season_names.get(season, season)}: {avg:.0f} bicikala")

    # Vizuelizacije
    try:
        # Korelaciona matrica
        plt.figure(figsize=(16, 6))
        plt.subplot(1, 2, 1)
        mask = np.triu(np.ones_like(numeric_df.corr(), dtype=bool))
        sns.heatmap(numeric_df.corr(), mask=mask, annot=True, cmap='coolwarm', center=0, 
                   fmt='.2f', annot_kws={'size': 10}, square=True, linewidths=0.5)
        plt.title('Korelaciona matrica', fontsize=16, fontweight='bold')
        plt.xticks(fontsize=11, rotation=45, ha='right')
        plt.yticks(fontsize=11, rotation=0)

        # Boxplot - povećan i čitljiviji
        plt.subplot(1, 2, 2)
        available_cols = [col for col in ['temp', 'hum', 'windspeed', 'cnt'] if col in df.columns]
        df[available_cols].boxplot(figsize=(8, 6), fontsize=11)
        plt.title('Boxplot za detekciju outliera', fontsize=16, fontweight='bold')
        plt.ylabel('Vrednost', fontsize=12)
        plt.xlabel('Promenljive', fontsize=12)
        plt.xticks(fontsize=11, rotation=0)
        plt.yticks(fontsize=10)

        plt.tight_layout()
        plt.savefig('korelacija_i_anomalije.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Vizuelizacije sačuvane u 'korelacija_i_anomalije.png'")
    except Exception as e:
        print(f"Greška pri kreiranju vizuelizacija: {e}")

def feature_engineering(df):
    """
    Kreira nove karakteristike (features) iz postojećih podataka.
    Args:
        df: DataFrame sa podacima
    Returns:
        DataFrame: Dataset sa novim karakteristikama
    """
    print(" Kreiranje novih karakteristika...")

    # Ciklične transformacije za sat
    if 'hr' in df.columns:
        df['hr_sin'] = np.sin(2 * np.pi * df['hr'] / 24)
        df['hr_cos'] = np.cos(2 * np.pi * df['hr'] / 24)
        print("Dodato: hr_sin, hr_cos")

    # Ciklične transformacije za mesec
    if 'mnth' in df.columns:
        df['mnth_sin'] = np.sin(2 * np.pi * (df['mnth'] - 1) / 12)
        df['mnth_cos'] = np.cos(2 * np.pi * (df['mnth'] - 1) / 12)
        print("Dodato: mnth_sin, mnth_cos")

    # Rush hour
    if 'hr' in df.columns:
        df['rush_hour'] = ((df['hr'].between(7, 9)) | (df['hr'].between(17, 19))).astype(int)
        print("Dodato: rush_hour")

    # Temperature kategorije
    if 'temp' in df.columns:
        df['temp_category'] = pd.cut(df['temp'], bins=[0, 0.3, 0.7, 1.0], labels=[0, 1, 2], include_lowest=True).astype(int)
        print("Dodato: temp_category")

    return df

def encode_categorical(df):
    """
    Enkodira kategorijalne varijable koristeći OneHotEncoder.
    Args:
        df: DataFrame sa podacima
    Returns:
        DataFrame: Enkodirani dataset
    """
    print(" Enkodiranje kategorijalnih varijabli...")

    categorical_cols = ['season', 'weathersit', 'weekday']
    available_categorical = [col for col in categorical_cols if col in df.columns]  # Provera da li kolone postoje

    if available_categorical:
        try:
            encoder = OneHotEncoder(drop='first', sparse_output=False)  # sparse_output=False vraća obican NumPy array umesto matrice
            encoded_cols = encoder.fit_transform(df[available_categorical])
            encoded_df = pd.DataFrame(encoded_cols, 
                                    columns=encoder.get_feature_names_out(available_categorical),  # Pravi dataFrame sa imenima kolona (generisanim od strane OneHotEncoder) i ispunjava enkodirane vrednosti
                                    index=df.index)

            df = df.drop(available_categorical, axis=1) # Ukloni originalne kategorijalne kolone
            df = pd.concat([df, encoded_df], axis=1)    # Dodaj enkodirane kolone nazad u glavni DataFrame koje popunjavaju mesta originalnih kolona
            print(f"Dodato {encoded_cols.shape[1]} enkodiranih kolona")
        except Exception as e:
            print(f"Greška pri enkodiranju: {e}")

    return df

def prepare_data(df):
    """
    Priprema podatke za treniranje modela - TEMPORAL SPLIT.
    Args:
        df: DataFrame sa podacima
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    print(" Priprema za treniranje (temporal split)...")

    # Temporal split HRONOLOSKI, da buduci rezultati ne bi predvideli proslost, jer ne znamo da li je sistem zavistan od vremena ili ne, pa pretpostavljamo da jeste 
    #  Training / Test
    # (80% train, 20% test)

    df = df.sort_values('dteday')
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()  # Nalazi granicu za 80% podataka i deli DataFrame na train i test delove na osnovu te granice
    test_df = df.iloc[split_idx:].copy()

    print(f"Training period: {train_df['dteday'].min()} do {train_df['dteday'].max()}")
    print(f"Test period: {test_df['dteday'].min()} do {test_df['dteday'].max()}")

    train_df = train_df.drop('dteday', axis=1)      # Ukloni dteday kolonu jer nije potrebna za treniranje
    test_df = test_df.drop('dteday', axis=1)    


    X_train = train_df.drop('cnt', axis=1)
    X_test = test_df.drop('cnt', axis=1)
    y_train = train_df['cnt']
    y_test = test_df['cnt']

    # Ukloni redundantne kolone
    redundant_cols = ['hr', 'mnth'] if 'hr_sin' in X_train.columns else []
    for col in redundant_cols:
        if col in X_train.columns:
            X_train = X_train.drop(col, axis=1)
            X_test = X_test.drop(col, axis=1)
            print(f"Uklonjena redundantna '{col}' kolona")

    # Popuni missing values
    numeric_means = X_train.select_dtypes(include=[np.number]).mean()
    X_train = X_train.fillna(numeric_means)
    X_test = X_test.fillna(numeric_means)

    # Log transformacija
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)

    print(f"Features: {X_train.shape[1]} kolona")
    print(f"Training samples: {X_train.shape[0]} redova")
    print(f"Test samples: {X_test.shape[0]} redova")

    return X_train, X_test, y_train_log, y_test_log

# Ovde se nalazi i cross validation
def train_model(X_train, y_train):
    """
    Trenira i optimizuje više modela.
    Args:
        X_train: Training features
        y_train: Training target
    Returns:
        dict: Istrenirani modeli i parametri
    """
    print(" Treniranje i optimizacija modela...")
    trained_models = {}

    # Osnovno treniranje različitih algoritama. Moze postojati i npr. linear regression , ali je dodatan posao, a ocigledno je da ne moze da prismridi RFR-u, a pogotovo tek GBR-u.
    # Istestirao sam, znam da ne valja, imao sam neke probleme sa ispisom, pa sam ga se resio.

    base_models = {
        'RandomForest': RandomForestRegressor(n_estimators=50, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=50, random_state=42)
    }

    print("Osnovno testiranje algoritama:")

    for name, model in base_models.items():
        model.fit(X_train, y_train)
        cv_scores = cross_val_score(model, X_train, np.expm1(y_train), cv=3, scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(-cv_scores.mean())
        print(f"  {name}: CV RMSE = {cv_rmse:.1f}")

    # Grid Search optimizacija za Random Forest
    print("\nGrid Search optimizacija za Random Forest...")
    rf_params = {
        'n_estimators': [50, 100],
        'max_depth': [8, 10],
        'min_samples_split': [5, 10]
    }

    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    rf_grid = GridSearchCV(rf, rf_params, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)  # Moglo je i neg_mean_absolute_error, uobičajeno se koristi MSE
    rf_grid.fit(X_train, y_train)                                                           # Ovde se desava unakrsna validacija i pronalaze najbolji parametri

    print(f"Najbolji RF parametri: {rf_grid.best_params_}")

    # Grid Search optimizacija za Gradient Boosting
    print("Grid Search optimizacija za Gradient Boosting...")
    gb_params = {
        'n_estimators': [50, 100],
        'max_depth': [6, 8],
        'learning_rate': [0.1, 0.05]
    }

    gb = GradientBoostingRegressor(random_state=42)                                                    
    gb_grid = GridSearchCV(gb, gb_params, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)   # Moglo je i neg_mean_absolute_error, uobičajeno se koristi MSE      
    gb_grid.fit(X_train, y_train)                                                            # Ovde se desava unakrsna validacija i pronalaze najbolji parametri

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

    print(f"Najbolji GB parametri: {rf_grid.best_params_}")

    return trained_models

def evaluate_model(models, X_train, X_test, y_train, y_test, model_name):
    """
    Evaluira performanse modela.
    Args:
        models: Rečnik istreniranih modela
        X_train, X_test: Training i test features
        y_train, y_test: Training i test target (u log skali)
        model_name: Naziv modela
    Returns:
        dict: Metrike performansi
    """
    model = models[model_name]['model']

    # Predikcije
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Konvertuj nazad iz log skale
    y_train_orig = np.expm1(y_train)
    y_test_orig = np.expm1(y_test)
    y_pred_train_orig = np.expm1(y_pred_train)
    y_pred_test_orig = np.expm1(y_pred_test)

    # Izračunaj metrike
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
    Prikazuje važnost karakteristika za najbolji model.
    Args:
        trained_models: Rečnik istreniranih modela
        feature_names: Nazivi karakteristika
    Returns:
        DataFrame: Feature importance tabela
    """
    print("\n Analiza važnosti karakteristika...")

    # Uzmi najbolji model
    best_model_name = list(trained_models.keys())[1]
    model = trained_models[best_model_name]['model']

    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("Top 10 najbitnijih features:")
    for _, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.3f}")

    return feature_importance

def feature_selection_comparison(trained_models, feature_importance, X_train, X_test, y_train, y_test):
    """
    Poredi performanse sa različitim brojem features.
    """
    print("\n Poređenje sa različitim brojem features...")

    best_model_name = list(trained_models.keys())[0]
    best_params = trained_models[best_model_name]['params']

    feature_counts = [5, 10, len(X_train.columns)]
    results = {}

    for n_features in feature_counts:
        top_features = feature_importance.head(n_features)['feature'].tolist()
        X_train_sel = X_train[top_features]
        X_test_sel = X_test[top_features]
        label = f"Top {n_features}"

        # Treniraj model sa selektovanim features
        model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
        model.fit(X_train_sel, y_train)
        y_pred = model.predict(X_test_sel)

        # Konvertuj iz log skale
        y_test_orig = np.expm1(y_test)
        y_pred_orig = np.expm1(y_pred)
        r2 = r2_score(y_test_orig, y_pred_orig)

        results[label] = r2
        print(f"  {label} features: R² = {r2:.3f}")

    return results

def show_sample_predictions(y_test, y_pred_test, n_samples=5):
    """
    Prikazuje primere predikcija.
    """
    print("\n Primeri predikcija...")

    y_test_orig = np.expm1(y_test)
    y_pred_test_orig = np.expm1(y_pred_test)

    n_samples = min(n_samples, len(y_test))
    sample_indices = np.random.choice(len(y_test), n_samples, replace=False)

    print(f"{'#':<3} {'Stvarno':<10} {'Predviđeno':<12} {'Greška'}")
    print("-" * 35)

    for i, idx in enumerate(sample_indices):
        actual = int(y_test_orig.iloc[idx])
        predicted = int(y_pred_test_orig[idx])
        error = abs(actual - predicted)
        print(f"{i+1:<3} {actual:<10} {predicted:<12} {error}")

def main():
    """
    Glavna funkcija programa.
    """
    print("=== BIKE SHARING DEMAND PREDICTOR ===\n")

    try:
        # Učitaj dataset
        df = load_dataset()
        if df is None:
            return None

        # Očisti podatke
        df = clean_data(df)
        if df is None:
            return None

        # Eksplorativna analiza
        exploratory_data_analysis(df.copy())

        # Feature engineering
        df = feature_engineering(df)

        # Enkodiranje
        df = encode_categorical(df)

        # Pripremi podatke
        X_train, X_test, y_train, y_test = prepare_data(df)

        # Treniraj modele
        trained_models = train_model(X_train, y_train)

        # Evaluacija najboljih modela
        print("\nEvaluacija najboljih modela...")
        results = {}

        # Iterira kroz OBA modela!
        for name in trained_models:
            results[name] = evaluate_model(trained_models, X_train, X_test, y_train, y_test, name)
            
            # Analiza kvaliteta
            r2 = results[name]['test_r2']
            if r2 > 0.8:
                quality = "ODLIČAN"
            elif r2 > 0.7:
                quality = "DOBAR"
            else:
                quality = "ZADOVOLJAVAJUĆI"

            overfitting = results[name]['overfitting']
            stability = "stabilan" if overfitting < 0.1 else "overfitting"
            
            print(f"{name}:")
            print(f"  Test R²: {r2:.3f} ({quality})")
            print(f"  Test RMSE: {results[name]['test_rmse']:.1f}")
            print(f"  Test MAE: {results[name]['test_mae']:.1f}")
            print(f"  Stabilnost: {stability}")
        # Feature importance
        feature_importance = show_feature_importance(trained_models, X_train.columns)

        # Feature selection poređenje
        fs_results = feature_selection_comparison(trained_models, feature_importance, 
                                                X_train, X_test, y_train, y_test)

        # Primeri predikcija
        best_model_name = max(results.keys(), key=lambda k: results[k]['test_r2'])
        best_model = trained_models[best_model_name]['model']
        y_pred_test = best_model.predict(X_test)
        show_sample_predictions(y_test, y_pred_test)

        best_r2 = results[best_model_name]['test_r2']
        best_rmse = results[best_model_name]['test_rmse']


        print(f"\nNajbolji model: {best_model_name}")
        print(f"Test R²: {best_r2:.3f}")
        print(f"Test RMSE: {best_rmse:.1f} bicikala")

        quality = "ODLIČAN" if best_r2 > 0.8 else "DOBAR" if best_r2 > 0.7 else "ZADOVOLJAVAJUĆI"
        print(f"Kvalitet: {quality}")

        print("=" * 60)

        return trained_models, results, feature_importance

    except Exception as e:
        print(f"\nGreška u programu: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = main()
    if result is not None:
        print("\nProgram uspešno završen!")
    else:
        print("\nProgram nije mogao da se izvrši.")
