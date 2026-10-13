
# ============================================================
# PROJECT: SABER 11 STUDENT PERFORMANCE PREDICTION
# Task: Supervised Learning - Regression
# Dataset: Colombian ICFES Saber 11, 2020-2
# ============================================================

# 1. IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

sns.set_theme(style="whitegrid")
np.random.seed(42)

# ============================================================
# 2. LOAD DATA
# ============================================================

DATA_PATH = "Saber_11__2020-2_20261004.csv"

features = [
    "ESTU_GENERO",
    "FAMI_ESTRATOVIVIENDA",
    "FAMI_EDUCACIONPADRE",
    "FAMI_EDUCACIONMADRE",
    "FAMI_TIENEINTERNET",
    "FAMI_TIENECOMPUTADOR",
    "FAMI_TIENEAUTOMOVIL",
    "FAMI_PERSONASHOGAR",
    "FAMI_NUMLIBROS",
    "FAMI_SITUACIONECONOMICA",
    "COLE_NATURALEZA",
    "COLE_AREA_UBICACION",
    "COLE_JORNADA",
    "COLE_CALENDARIO",
    "COLE_BILINGUE",
    "COLE_CARACTER",
    "ESTU_DEDICACIONLECTURADIARIA",
    "ESTU_DEDICACIONINTERNET",
    "ESTU_HORASSEMANATRABAJA"
]

target = "PUNT_GLOBAL"
required_columns = features + [target]

df = pd.read_csv(
    DATA_PATH,
    usecols=lambda column: column in required_columns,
    low_memory=False
)

print("Original dataset shape:", df.shape)

# Use a reproducible sample for faster experimentation
if len(df) > 50000:
    df = df.sample(n=50000, random_state=42)

df = df.dropna(subset=[target]).copy()

print("Working dataset shape:", df.shape)
display(df.head())

# ============================================================
# 3. EXPLORATORY DATA ANALYSIS
# ============================================================

print("Dataset information:")
df.info()

print("\nTarget variable statistics:")
display(df[target].describe().round(2))

print("\nMissing values (%):")
missing_percent = (
    df[features].isnull().mean().sort_values(ascending=False) * 100
)
display(missing_percent.round(2))

print("\nDuplicate rows:", df.duplicated().sum())

# Distribution of global scores
plt.figure(figsize=(9, 5))
sns.histplot(df[target], bins=40, kde=True)
plt.title("Distribution of Saber 11 Global Scores")
plt.xlabel("Global Score")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.show()

# Scores by socioeconomic stratum
plt.figure(figsize=(10, 5))
sns.boxplot(
    data=df,
    x="FAMI_ESTRATOVIVIENDA",
    y=target
)
plt.title("Global Score by Household Socioeconomic Stratum")
plt.xlabel("Socioeconomic Stratum")
plt.ylabel("Global Score")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# Scores by school type, internet access, and location
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

comparison_features = [
    "COLE_NATURALEZA",
    "FAMI_TIENEINTERNET",
    "COLE_AREA_UBICACION"
]

for ax, feature in zip(axes, comparison_features):
    sns.boxplot(
        data=df,
        x=feature,
        y=target,
        ax=ax
    )
    ax.set_title(f"Global Score by {feature}")
    ax.tick_params(axis="x", rotation=25)

plt.tight_layout()
plt.show()

# Compare average scores across selected categories
for feature in [
    "FAMI_ESTRATOVIVIENDA",
    "FAMI_TIENEINTERNET",
    "COLE_NATURALEZA",
    "COLE_JORNADA"
]:
    print(f"\nAverage score by {feature}:")
    display(
        df.groupby(feature, dropna=False)[target]
        .agg(["count", "mean"])
        .round(2)
    )

# ============================================================
# 4. DATA PREPROCESSING
# ============================================================

# Remove exact duplicate rows
df = df.drop_duplicates().copy()

# Separate predictors and target
X = df[features].copy()
y = df[target].copy()

# Treat categorical variables as text and missing values
# as an explicit category
X = X.fillna("Missing").astype(str)

print("Feature matrix shape:", X.shape)
print("Target vector shape:", y.shape)

# Split data before fitting the encoder to avoid data leakage
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# One-hot encode categorical variables.
# The encoder is fitted within each model pipeline.
encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

# ============================================================
# 5. DEFINE REGRESSION MODELS
# ============================================================

models = {
    "Linear Regression": make_pipeline(
        encoder,
        LinearRegression()
    ),

    "Random Forest": make_pipeline(
        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
        RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=42
        )
    ),

    "Gradient Boosting": make_pipeline(
        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
        HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.1,
            random_state=42
        )
    )
}

# ============================================================
# 6. TRAIN AND EVALUATE MODELS
# ============================================================

results = []
predictions = {}
fitted_models = {}

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    predicted_scores = model.predict(X_test)

    predictions[name] = predicted_scores
    fitted_models[name] = model

    # Cross-validation uses training data only
    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=3,
        scoring="r2",
        n_jobs=-1
    )

    results.append({
        "Model": name,
        "MAE": mean_absolute_error(y_test, predicted_scores),
        "RMSE": np.sqrt(
            mean_squared_error(y_test, predicted_scores)
        ),
        "R2": r2_score(y_test, predicted_scores),
        "Cross-Validation R2": cv_scores.mean()
    })

    print("MAE:", round(
        mean_absolute_error(y_test, predicted_scores), 2
    ))
    print("RMSE:", round(
        np.sqrt(mean_squared_error(y_test, predicted_scores)), 2
    ))
    print("R2:", round(
        r2_score(y_test, predicted_scores), 3
    ))
    print("Mean CV R2:", round(cv_scores.mean(), 3))

# ============================================================
# 7. MODEL COMPARISON
# ============================================================

results_table = (
    pd.DataFrame(results)
    .set_index("Model")
    .sort_values("R2", ascending=False)
)

print("\nModel comparison:")
display(results_table.round(3))

# Select the model with the highest test R2
best_model_name = results_table["R2"].idxmax()
best_model = fitted_models[best_model_name]

print("Best model based on test R2:", best_model_name)

# ============================================================
# 8. ACTUAL VS. PREDICTED SCORES
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, (name, predicted_scores) in zip(
    axes, predictions.items()
):
    ax.scatter(
        y_test,
        predicted_scores,
        alpha=0.2,
        s=10
    )

    minimum = min(y_test.min(), predicted_scores.min())
    maximum = max(y_test.max(), predicted_scores.max())

    ax.plot(
        [minimum, maximum],
        [minimum, maximum],
        color="red",
        linestyle="--"
    )

    ax.set_title(name)
    ax.set_xlabel("Actual Global Score")
    ax.set_ylabel("Predicted Global Score")

plt.suptitle("Actual vs. Predicted Saber 11 Scores")
plt.tight_layout()
plt.show()

# ============================================================
# 9. ERROR ANALYSIS
# ============================================================

best_predictions = predictions[best_model_name]
errors = y_test - best_predictions

plt.figure(figsize=(9, 5))
sns.histplot(errors, bins=40, kde=True)
plt.axvline(0, color="red", linestyle="--")
plt.title(f"Prediction Errors - {best_model_name}")
plt.xlabel("Actual Score - Predicted Score")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.show()

# Compare MAE and RMSE
results_table[["MAE", "RMSE"]].plot(
    kind="bar",
    figsize=(9, 5)
)

plt.title("Prediction Error by Model")
plt.ylabel("Error in Score Points")
plt.xlabel("")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# ============================================================
# 10. FEATURE IMPORTANCE
# ============================================================

# Use a reproducible subset to reduce computation time
sample_size = min(2000, len(X_test))
importance_sample = X_test.sample(
    n=sample_size,
    random_state=42
)

importance_target = y_test.loc[importance_sample.index]

importance_result = permutation_importance(
    best_model,
    importance_sample,
    importance_target,
    n_repeats=3,
    random_state=42,
    scoring="r2",
    n_jobs=-1
)

feature_importance = pd.Series(
    importance_result.importances_mean,
    index=X.columns
).sort_values(ascending=False)

print("\nTop 10 most influential features:")
display(feature_importance.head(10).round(4))

feature_importance.head(10).sort_values().plot(
    kind="barh",
    figsize=(9, 6)
)

plt.title(f"Top 10 Features - {best_model_name}")
plt.xlabel("Decrease in R2 When Feature Is Permuted")
plt.tight_layout()
plt.show()

# ============================================================
# 11. FINAL SUMMARY
# ============================================================

print("FINAL PROJECT SUMMARY")
print("=" * 50)
print("Best model:", best_model_name)
print("MAE:", round(results_table.loc[best_model_name, "MAE"], 2))
print("RMSE:", round(results_table.loc[best_model_name, "RMSE"], 2))
print("R2:", round(results_table.loc[best_model_name, "R2"], 3))
print(
    "Cross-validation R2:",
    round(
        results_table.loc[best_model_name, "Cross-Validation R2"],
        3
    )
)

print("\nProject completed successfully.")
