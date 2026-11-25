import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)


def load_or_generate_data(csv_path: str = "bankruptcy_data.csv") -> pd.DataFrame:
    """
    Loads company financial data from CSV if available; otherwise generates a
    realistic synthetic dataset for demonstration and testing.
    Expected target column: 'bankruptcy' (1 = Bankruptcy Risk, 0 = Non-Bankruptcy).
    """
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded dataset from '{csv_path}' with shape {df.shape}.")
        return df
    except FileNotFoundError:
        print(f"'{csv_path}' not found. Generating synthetic financial dataset...")
        rng = np.random.default_rng(42)
        n_samples = 1000

        # Financial ratios / risk indicators
        industrial_risk = rng.choice(["low", "medium", "high"], size=n_samples, p=[0.4, 0.35, 0.25])
        management_risk = rng.choice(["low", "medium", "high"], size=n_samples, p=[0.4, 0.35, 0.25])
        financial_flexibility = rng.normal(loc=0.5, scale=0.35, size=n_samples)
        credibility = rng.normal(loc=0.5, scale=0.35, size=n_samples)
        competitiveness = rng.normal(loc=0.5, scale=0.35, size=n_samples)
        operating_risk = rng.normal(loc=0.5, scale=0.35, size=n_samples)

        # Risk score where lower flexibility/credibility/competitiveness increases bankruptcy risk
        risk_score = (
            -1.4 * financial_flexibility
            - 1.3 * credibility
            - 1.5 * competitiveness
            + 0.9 * operating_risk
            + np.where(industrial_risk == "high", 0.6, np.where(industrial_risk == "medium", 0.2, -0.2))
            + np.where(management_risk == "high", 0.6, np.where(management_risk == "medium", 0.2, -0.2))
            + rng.normal(0, 0.4, size=n_samples)
        )
        prob = 1 / (1 + np.exp(-risk_score))
        bankruptcy = (prob > 0.5).astype(int)

        df = pd.DataFrame({
            "industrial_risk": industrial_risk,
            "management_risk": management_risk,
            "financial_flexibility": np.round(financial_flexibility, 3),
            "credibility": np.round(credibility, 3),
            "competitiveness": np.round(competitiveness, 3),
            "operating_risk": np.round(operating_risk, 3),
            "bankruptcy": bankruptcy,
        })

        # Inject a small fraction of missing values to demonstrate preprocessing
        for col in ["financial_flexibility", "credibility", "industrial_risk"]:
            missing_idx = rng.choice(n_samples, size=20, replace=False)
            df.loc[missing_idx, col] = np.nan

        return df


def perform_eda(df: pd.DataFrame, target_col: str = "bankruptcy") -> None:
    """Runs basic Exploratory Data Analysis (EDA) using Pandas."""
    print("\n--- Exploratory Data Analysis (EDA) ---")
    print("First 5 rows:")
    print(df.head())

    print("\nDataset Info & Missing Values:")
    print(df.info())
    print("\nMissing values per column:\n", df.isnull().sum())

    print("\nTarget Class Distribution (0 = Healthy/Non-Bankrupt, 1 = Bankruptcy Risk):")
    print(df[target_col].value_counts(dropna=False))
    print(df[target_col].value_counts(normalize=True).round(4))

    numeric_cols = df.select_dtypes(include=[np.number]).columns.drop(target_col, errors="ignore")
    if len(numeric_cols) > 0:
        print("\nSummary Statistics (Numeric Features):")
        print(df[numeric_cols].describe().T[["mean", "std", "min", "50%", "max"]])


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Creates preprocessing pipelines for numerical and categorical features."""
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor


def evaluate_model(name: str, model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Evaluates a trained pipeline and prints key classification metrics."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print(f"\n=== {name} Evaluation ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred, digits=4))

    return {"Model": name, "Accuracy": acc, "ROC_AUC": roc_auc}


def main():
    target_col = "bankruptcy"
    df = load_or_generate_data("bankruptcy_data.csv")

    # Clean duplicate rows if any
    initial_rows = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    if len(df) < initial_rows:
        print(f"Removed {initial_rows - len(df)} duplicate rows.")

    # Ensure target is integer (0/1) and drop rows with missing target
    df = df.dropna(subset=[target_col]).copy()
    df[target_col] = df[target_col].astype(int)

    perform_eda(df, target_col=target_col)

    # Split features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Stratified train-test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor(X)

    # 1. Logistic Regression Pipeline
    log_reg_clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ])

    # 2. Decision Tree Pipeline
    dt_clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=42)),
    ])

    # Train models
    log_reg_clf.fit(X_train, y_train)
    dt_clf.fit(X_train, y_train)

    # Evaluate models
    results = []
    results.append(evaluate_model("Logistic Regression", log_reg_clf, X_test, y_test))
    results.append(evaluate_model("Decision Tree", dt_clf, X_test, y_test))

    comparison_df = pd.DataFrame(results)
    print("\n--- Model Comparison Summary ---")
    print(comparison_df.to_string(index=False))

    # Predict on a sample new company record
    sample_company = pd.DataFrame([X_test.iloc[0].to_dict()])
    print("\n--- Sample Company Prediction ---")
    print("Input features:\n", sample_company.to_string(index=False))
    print(f"Actual label       : {y_test.iloc[0]}")
    print(f"Logistic Regression: {log_reg_clf.predict(sample_company)[0]} "
          f"(prob risk = {log_reg_clf.predict_proba(sample_company)[0, 1]:.4f})")
    print(f"Decision Tree      : {dt_clf.predict(sample_company)[0]} "
          f"(prob risk = {dt_clf.predict_proba(sample_company)[0, 1]:.4f})")


if __name__ == "__main__":
    main()