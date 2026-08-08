from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_PATH = Path("data/breast_cancer.csv")

mlflow.set_tracking_uri("sqlite:///mlflow.db")

mlflow.set_experiment("Breast Cancer Classification")

# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print(f"Dataset shape: {df.shape}")
print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")


# ============================================================
# 3. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop("target", axis=1)
y = df["target"]

print("\n" + "=" * 60)
print("FEATURES AND TARGET")
print("=" * 60)

print(f"Number of features: {X.shape[1]}")
print("Target column: target")


# ============================================================
# 4. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n" + "=" * 60)
print("TRAIN-TEST SPLIT")
print("=" * 60)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")


# ============================================================
# 5. DEFINE MODELS
# ============================================================

models = {

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=5000))
    ]),

    "Random Forest": Pipeline([
        ("model", RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ))
    ]),

    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC())
    ])
}


# ============================================================
# 6. TRAIN + MLFLOW TRACKING
# ============================================================

results = []
run_ids = {}

print("\n" + "=" * 60)
print("MODEL TRAINING + MLFLOW TRACKING")
print("=" * 60)

for model_name, model in models.items():

    print(f"\nTraining: {model_name}")

    # Start one MLflow run for this model
    with mlflow.start_run(run_name=model_name) as run:
        run_ids[model_name] = run.info.run_id

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        model.fit(X_train, y_train)

        # ----------------------------------------------------
        # Predict
        # ----------------------------------------------------

        y_pred = model.predict(X_test)

        # ----------------------------------------------------
        # Calculate metrics
        # ----------------------------------------------------

        accuracy = accuracy_score(y_test, y_pred)

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )

        # ----------------------------------------------------
        # Log common parameters
        # ----------------------------------------------------

        mlflow.log_param("model_name", model_name)
        mlflow.log_param("test_size", 0.20)
        mlflow.log_param("random_state", 42)

        # ----------------------------------------------------
        # Log model-specific parameters
        # ----------------------------------------------------

        if model_name == "Logistic Regression":

            mlflow.log_param("max_iter", 5000)
            mlflow.log_param("scaler", "StandardScaler")

        elif model_name == "Random Forest":

            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("random_state_model", 42)

        elif model_name == "SVM":

            mlflow.log_param("kernel", "rbf")
            mlflow.log_param("scaler", "StandardScaler")

        # ----------------------------------------------------
        # Log metrics
        # ----------------------------------------------------

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # ----------------------------------------------------
        # Log model artifact
        # ----------------------------------------------------

        mlflow.sklearn.log_model(
            model,
            name="model"
        )

        # ----------------------------------------------------
        # Store results
        # ----------------------------------------------------

        results.append({
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        })

        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")


# ============================================================
# 7. COMPARE MODELS
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by=["F1 Score", "Accuracy"],
    ascending=False
)

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 8. IDENTIFY BEST MODEL
# ============================================================

best_model_name = results_df.iloc[0]["Model"]

print("\n" + "=" * 60)
print("BEST MODEL")
print("=" * 60)

print(f"Best model: {best_model_name}")

print(
    f"Best F1 Score: "
    f"{results_df.iloc[0]['F1 Score']:.4f}"
)


# ============================================================
# 9. BEST MODEL REPORT
# ============================================================

best_model = models[best_model_name]

best_predictions = best_model.predict(X_test)

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        best_predictions,
        zero_division=0
    )
)
# ============================================================
# 10. REGISTER BEST MODEL
# ============================================================

best_run_id = run_ids[best_model_name]

model_uri = f"runs:/{best_run_id}/model"

registered_model_name = "BreastCancerClassifier"

print("\n" + "=" * 60)
print("MODEL REGISTRATION")
print("=" * 60)

print(f"Registering: {best_model_name}")
print(f"Run ID: {best_run_id}")

model_version = mlflow.register_model(
    model_uri=model_uri,
    name=registered_model_name
)

print(f"Registered model: {registered_model_name}")
print(f"Model version: {model_version.version}")