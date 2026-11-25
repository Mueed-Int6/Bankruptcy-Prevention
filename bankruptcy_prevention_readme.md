# Bankruptcy Prevention Using Machine Learning

A machine learning classification project built with **Python**, **Pandas**, and **Scikit-learn** to predict whether a company is at risk of bankruptcy based on financial and operational risk indicators. The project implements end-to-end data cleaning, exploratory data analysis (EDA), preprocessing pipelines, and comparative evaluation using **Logistic Regression** and **Decision Tree** classifiers.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Tech Stack](#tech-stack)
- [Dataset Schema](#dataset-schema)
- [Project Workflow](#project-workflow)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Model Evaluation](#model-evaluation)
- [Future Improvements](#future-improvements)

---

## Project Overview
Early identification of financial distress helps stakeholders, investors, and management take preventive measures before a company faces insolvency. This project frames bankruptcy prediction as a binary classification problem:
- **`0` (Non-Bankruptcy):** Company is financially stable.
- **`1` (Bankruptcy Risk):** Company exhibits significant risk of bankruptcy.

---

## Tech Stack
- **Language:** Python 3.8+
- **Data Manipulation & Analysis:** Pandas, NumPy
- **Visualization:** Matplotlib
- **Machine Learning:** Scikit-learn (`LogisticRegression`, `DecisionTreeClassifier`, `ColumnTransformer`, `Pipeline`, `SimpleImputer`, `StandardScaler`, `OneHotEncoder`)
- **Environment:** Jupyter Notebook / Google Colab / VS Code

---

## Dataset Schema
The model expects a CSV dataset named `bankruptcy_data.csv` with the following columns (if no CSV is provided, the script automatically generates a synthetic dataset with the same structure for testing):

| Feature | Type | Description |
| :--- | :--- | :--- |
| `industrial_risk` | Categorical (`low`, `medium`, `high`) | Level of external risk in the company's industry sector |
| `management_risk` | Categorical (`low`, `medium`, `high`) | Risk associated with corporate governance and leadership |
| `financial_flexibility` | Numerical | Company's ability to respond to unexpected financial needs |
| `credibility` | Numerical | Creditworthiness and trust among lenders/stakeholders |
| `competitiveness` | Numerical | Market strength relative to industry peers |
| `operating_risk` | Numerical | Risk arising from day-to-day business operations |
| `bankruptcy` *(Target)* | Binary (`0` or `1`) | Target label (`0` = Healthy, `1` = At Risk of Bankruptcy) |

---

## Project Workflow
1. **Data Loading & Cleaning:**
   - Loads `bankruptcy_data.csv` (or falls back to synthetic data generation).
   - Removes duplicate records and drops rows with missing target labels.
2. **Exploratory Data Analysis (EDA):**
   - Inspects dataset dimensions, data types, missing values, class balance, and summary statistics using Pandas.
3. **Data Preprocessing (`ColumnTransformer` & `Pipeline`):**
   - **Numerical Features:** Median imputation (`SimpleImputer`) followed by feature standardization (`StandardScaler`).
   - **Categorical Features:** Most-frequent imputation followed by one-hot encoding (`OneHotEncoder`).
   - Uses an **80/20 stratified train-test split** to preserve class proportions.
4. **Model Training & Comparison:**
   - Trains a **Logistic Regression** classifier as a linear baseline.
   - Trains a regularized **Decision Tree** classifier (`max_depth=5`, `min_samples_leaf=10`) to capture non-linear decision rules without overfitting.
5. **Evaluation & Inference:**
   - Compares models using **Accuracy**, **ROC-AUC**, **Confusion Matrix**, and **Classification Report** (Precision, Recall, F1-Score).
   - Runs sample inference on unseen company records with predicted probabilities.

---

## Installation & Setup

1. **Clone the repository or download the project files:**
   ```bash
   git clone https://github.com/your-username/bankruptcy-prevention-ml.git
   cd bankruptcy-prevention-ml
   ```

2. **Install required dependencies:**
   ```bash
   pip install numpy pandas matplotlib scikit-learn
   ```

---

## Usage

1. Place your dataset as `bankruptcy_data.csv` in the root directory (or let the script generate sample data automatically).
2. Run the main training and evaluation script:
   ```bash
   python bankruptcy_prevention.py
   ```
3. Review the printed EDA summaries, classification metrics, model comparison table, and sample company risk predictions in the console.

---

## Model Evaluation
Both models are evaluated on the held-out 20% test set using:
- **Accuracy:** Overall proportion of correct predictions.
- **Precision & Recall:** Critical for minimizing false negatives (failing to flag an at-risk company).
- **F1-Score:** Harmonic mean of precision and recall.
- **ROC-AUC Score:** Measures the model's ability to distinguish between healthy and at-risk companies across classification thresholds.

---

## Future Improvements
- Add **ROC Curve** and **Feature Importance** visualizations using Matplotlib.
- Experiment with ensemble classifiers such as **Random Forest** and **Gradient Boosting (XGBoost / LightGBM)**.
- Apply hyperparameter tuning using `GridSearchCV` or `RandomizedSearchCV`.
- Deploy the trained model pipeline as an interactive web app using **Streamlit** or **Flask**.

---

## Author
**Shaik Abdul Mueed**  
B.Tech - CSE (AI & ML) | G Pullaiah College of Engineering and Technology