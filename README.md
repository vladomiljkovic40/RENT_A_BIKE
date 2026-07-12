# 📊 Bike Sharing Demand Predictor

An end-to-end Machine Learning pipeline designed to analyze, engineer, and predict hourly bike rental demand using tree-based ensemble methods. This system processes raw environmental and seasonal data, handles temporal constraints, optimizes hyperparameters, and deploys the best-performing model automatically.

---

## 🏗️ Pipeline Architecture

The system is structured as a modular pipeline to guarantee clean data isolation, prevent data leakage, and ensure easy maintenance.

```
[ Raw Data: hour.csv ]
         │
         ▼
[ 1. Data Cleaning & Outlier Clipping ]
         │
         ▼
[ 2. Exploratory Data Analysis & Plots ]
         │
         ▼
[ 3. Feature Engineering & Time Transforms ]
         │
         ▼
[ 4. Categorical Encoding (One-Hot) ]
         │
         ▼
[ 5. Temporal Data Splitting (80/20) ]
         │
         ▼
[ 6. Hyperparameter Tuning (Grid Search) ]
         │
         ▼
[ 7. Evaluation & Feature Importance ]

```

---

## 🛠️ Core Features & Engineering Steps

### 1. Data Cleaning & Robust Outlier Handling

Instead of aggressively dropping rows with extreme values—which can cause substantial data loss—the system uses **Outlier Clipping**.

* It computes the Interquartile Range ($IQR$) via:

$$IQR = Q_3 - Q_1$$


* Boundaries are set at $Q_1 - 1.5 \times IQR$ and $Q_3 + 1.5 \times IQR$.
* Values falling outside these limits are capped to the boundary line, preserving data volume while neutralizing extreme variance. Missing values are filled dynamically using column means.

### 2. Feature Engineering & Cyclic Time Encoding

Hours (`0-23`) and months (`1-12`) are naturally cyclical. Standard numerical mapping forces a model to interpret hour `23` and hour `0` as opposites, even though they are consecutive.
To fix this, the script maps time features onto a 2D circle using sine and cosine transformations:

* $\text{hr\_sin} = \sin\left(\frac{2\pi \times \text{hr}}{24}\right)$
* $\text{hr\_cos} = \cos\left(\frac{2\pi \times \text{hr}}{24}\right)$

Additionally, domain-specific indicators like **Rush Hour** blocks (`7-9 AM` and `5-7 PM`) and discrete **Temperature Categories** are extracted to maximize predictive power.

### 3. Chronological (Temporal) Data Splitting

Standard random splits (`train_test_split`) introduce **data leakage** when handling time-series data because the model inadvertently uses future information to predict past events. This script enforces a strict chronological split: the first 80% of data forms the training set, and the final 20% forms the test set.

### 4. Target Scaling Optimization

Rental demand distributions (`cnt`) are heavily right-skewed. To prevent large demand spikes from skewing regression errors, the target variable undergoes a log transformation:


$$\tilde{y} = \ln(y + 1)$$


Predictions are automatically converted back to the original scale using the exponential function ($\text{expm1}$) before final metrics are calculated.

---

## 🤖 Model Optimization & Evaluation Strategy

The script evaluates two powerful ensemble techniques: **Random Forest Regressor** and **Gradient Boosting Regressor**.

### Grid Search Hyperparameter Tuning

To locate the absolute best configuration, the pipeline utilizes `GridSearchCV` combined with 3-fold cross-validation. It systematically tunes arrays of tree counts (`n_estimators`), tree depths (`max_depth`), and learning speeds (`learning_rate`).

### Key Performance Metrics

Models are judged based on three critical statistical anchors:

* **$R^2$ Score (Coefficient of Determination):** Measures the proportion of variance in demand explained by the input features. A score over `0.8` flags excellent fit.
* **RMSE (Root Mean Squared Error):** Quantifies the standard deviation of residuals, penalizing larger errors more heavily.
* **MAE (Mean Absolute Error):** Measures the average absolute error magnitude, telling us exactly how many bike units the predictions miss by on average.
* **Overfitting Tracking:** Evaluated by analyzing the absolute difference between training $R^2$ and testing $R^2$. A variance larger than `0.1` flags an overfit model.

---

## 📊 Feature Importance & Dimensionality Reduction

Once the top-performing model is determined, the script conducts a feature importance breakdown using **Mean Decrease in Impurity (MDI)** to show which features drive the splits.

Finally, a **Feature Selection Comparison** module tests the model using only the Top 5, Top 10, and all features. This reveals whether dropping minor variables can simplify the model and speed up inference times without sacrificing prediction accuracy.

---

## 🚀 Getting Started

### Prerequisites

Ensure you have the required dependencies installed:

```bash
pip install pandas numpy scikit-learn seaborn matplotlib

```

### Dataset Setup

Download the **Bike Sharing Dataset** (specifically the `hour.csv` file) and place it in the same directory as the main script.

### Running the Project

Execute the pipeline script:

```bash
python bike_predictor.py

```

### Outputs Generated

* **`correlation_and_outliers.png`**: High-resolution visualization showcasing the correlation matrix heatmap and outlier boxplots.
* **Console Logs**: Step-by-step printouts detailing data cleaning stats, cross-validation progress, best parameters, evaluation metrics, and sample live predictions.
