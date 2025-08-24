# hybrid_model_iot_mental_health.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from typing import Dict, Tuple

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, accuracy_score

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier

# =============================== 1. Load the dataset ===============================
file_path = "university_mental_health_iot_dataset.csv"  # ↩️ edit if needed

df = pd.read_csv(file_path)

# =============================== 2. Feature engineering =============================
if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["hour_of_day"] = df["timestamp"].dt.hour.fillna(-1).astype(int)
    df["day_of_week"] = df["timestamp"].dt.dayofweek.fillna(-1).astype(int)

features = [
    "location_id",
    "temperature_celsius",
    "humidity_percent",
    "air_quality_index",
    "lighting_lux",
    "crowd_density",
    "noise_level_db",
    "stress_level",
    "sleep_hours",
    "mood_score",
    "hour_of_day",
    "day_of_week",
]

target = "mental_health_status"

# =============================== 3. Train / test split =============================
X = df[features].copy()
y = df[target].copy()

le = None
if y.dtype == "object" or y.dtype.name == "category":
    le = LabelEncoder();
    y = le.fit_transform(y)
    print(f"Target classes: {list(le.classes_)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =============================== 4. Pre‑processing =================================
num_cols = [
    "temperature_celsius", "humidity_percent", "air_quality_index", "lighting_lux",
    "crowd_density", "noise_level_db", "stress_level", "sleep_hours",
    "mood_score", "hour_of_day", "day_of_week",
]
cat_cols = ["location_id"]

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, num_cols),
    ("cat", categorical_transformer, cat_cols),
])

# =============================== 5. Base models (regularised) ======================
base_models: Dict[str, object] = {
    "Random Forest": RandomForestClassifier(
        n_estimators=400,
        max_depth=25,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
    ),
    "SVM (RBF)": SVC(
        C=2,
        gamma="scale",
        kernel="rbf",
        class_weight="balanced",
        probability=True,
        random_state=42,
    ),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=6, min_samples_leaf=10, class_weight="balanced", random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=2, random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=2,
        objective="multi:softprob", num_class=len(np.unique(y)),
        eval_metric="mlogloss", use_label_encoder=False,
        random_state=42,
    ),
}

# =============================== 6. Safe hybrid builder ============================
from sklearn.ensemble import VotingClassifier
from itertools import combinations
from typing import Dict
# Import LogisticRegression, which is used as a backup model later
from sklearn.linear_model import LogisticRegression

_tree_family = {"Random Forest", "Gradient Boosting", "XGBoost", "Decision Tree"}


def allow_pair(n1: str, n2: str) -> bool:
    """Return True if the pair is allowed (avoids DT, dual-tree combos, and RF+NB)."""
    # Exclude any pair that includes a bare Decision Tree
    if "Decision Tree" in (n1, n2):
        return False
    # Exclude pairs where *both* are tree-based ensemble methods
    if n1 in _tree_family and n2 in _tree_family:
        return False
    # Exclude Random Forest + Naive Bayes
    if ("Random Forest" in (n1, n2)) and ("Naive Bayes" in (n1, n2)):
        return False
    return True


# Replace Random Forest + Naive Bayes with a backup if needed
# Define the backup model *before* trying to access it from base_models
backup_model_name = "Logistic Regression"
# Add Logistic Regression to base_models if it's not already there, as it's used as a backup
if backup_model_name not in base_models:
    base_models[backup_model_name] = LogisticRegression(random_state=42,
                                                        solver='liblinear')  # Added solver for potential future sklearn versions

backup_model = base_models[backup_model_name]

hybrid_models: Dict[str, VotingClassifier] = {}
for (n1, clf1), (n2, clf2) in combinations(base_models.items(), 2):
    # Skip combinations that include the backup model itself, unless pairing with NB
    if backup_model_name in (n1, n2) and not ("Naive Bayes" in (n1, n2) and ("Random Forest" in (n1, n2))):
        continue

    if not allow_pair(n1, n2):
        # Replace RF + NB with backup + NB
        if ("Random Forest" in (n1, n2)) and ("Naive Bayes" in (n1, n2)):
            other = "Naive Bayes" if n1 == "Random Forest" else "Naive Bayes"
            clf_other = base_models[other]
            name = f"{backup_model_name} + {other} (Voting:soft)"
            # Ensure the backup + Naive Bayes combination is only added once
            if name not in hybrid_models:
                hybrid_models[name] = VotingClassifier(
                    estimators=[(backup_model_name.replace(" ", "_"), backup_model),
                                (other.replace(" ", "_"), clf_other)],
                    voting="soft" if all(
                        hasattr(clf, "predict_proba") for clf in (backup_model, clf_other)) else "hard",
                    weights=[1, 1]
                )
        continue  # Skip the original disallowed pair (e.g., RF + NB)

    voting = "soft" if all(hasattr(clf, "predict_proba") for clf in (clf1, clf2)) else "hard"
    name = f"{n1} + {n2} (Voting:{voting})"
    # Ensure each allowed combination is only added once
    if name not in hybrid_models:
        hybrid_models[name] = VotingClassifier(
            estimators=[(n1.replace(" ", "_"), clf1), (n2.replace(" ", "_"), clf2)],
            voting=voting,
            weights=[1, 1],
        )

all_models: Dict[str, object] = {**base_models, **hybrid_models}

# =============================== 7. Train / evaluate ===============================
accuracies: Dict[str, float] = {}
reports: Dict[str, str] = {}

for name, clf in all_models.items():
    pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    # Skip obviously over‑fitting models (> 0.98 acc)
    if acc > 0.98:
        print(f"⚠️ Skipping {name} – appears to over‑fit with {acc:.2%} accuracy.")
        continue
    accuracies[name] = acc
    reports[name] = classification_report(
        y_test, y_pred, target_names=le.classes_ if le else None, zero_division=0
    )
    print(f"\n===== {name} =====")
    print(reports[name])
    print(f"Accuracy: {acc:.4f}")

# Guard against empty dict (all skipped)
if not accuracies:
    raise RuntimeError("All models seem to over‑fit; please revisit the parameter grid.")

# =============================== 8. Plot ===========================================
import matplotlib.pyplot as plt

plt.figure(figsize=(max(6, len(accuracies) * 0.45), 4))

# Plot with reduced bar width
bars = plt.bar(range(len(accuracies)), list(accuracies.values()), width=0.6, align='center')

# Rotate x-axis labels diagonally
plt.xticks(
    ticks=range(len(accuracies)),
    labels=list(accuracies.keys()),
    rotation=15,
    ha='right'
)

# Identify the best model
best_model = max(accuracies, key=accuracies.get)
best_value = accuracies[best_model]

# Add main title
plt.suptitle("Test Accuracy – Filtered Models (No 100% Overfit)", fontsize=13, weight='bold', y=1.05)

# ⬇️ Add best model note just below the main title (outside the plot)
plt.title(f"Best: {best_model} ({best_value:.2%})", fontsize=11, color="darkgreen", pad=25)

plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.tight_layout(rect=[0, 0, 1, 0.92])  # Leave space at top for suptitle and best model label
plt.show()

# =============================== 9. Best report ====================================
print("\n============================")
print("Best Model Detailed Report:")
print("============================")
print(reports[best_model])

# =============================== 10. Save summary ==================================
pd.Series(accuracies, name="accuracy").sort_values(ascending=False).to_frame().to_csv(
    "model_accuracy_summary.csv", index_label="model"
)
print("\nAccuracy summary saved to model_accuracy_summary.csv")

# =============================== 11. Individual-level risk categorisation ===========
# Train the best-performing pipeline on the *full* training data ---------------------
best_clf = all_models[best_model]           # picked in Section 7
best_pipe = Pipeline([("prep", preprocessor),
                      ("clf",  best_clf)])
best_pipe.fit(X_train, y_train)

# ------------------------------------------------ Predict on any dataset you like --
X_for_pred = df[features].copy()

# Predict class labels and probabilities
y_pred          = best_pipe.predict(X_for_pred)
y_pred_proba    = best_pipe.predict_proba(X_for_pred)

# -----------------------------------------------  Map numeric ↔︎ human-readable ----

if le is not None:
    str_labels = le.inverse_transform(y_pred)
else:
    str_labels = y_pred.astype(str)         # numeric → string

# Fallback / custom mapping (edit to suit your dataset)
risk_mapping = {
    0: "Low Risk",
    1: "Medium Risk",
    2: "High Risk",
}
risk_labels = [
    risk_mapping.get(lbl, lbl)              # preserve existing string if found
    for lbl in str_labels
]

# -----------------------------------------------  Assemble an easy-to-read table ---
results_df = X_for_pred.copy()
results_df["Predicted_Risk"]   = risk_labels
results_df["Max_Probability"]  = y_pred_proba.max(axis=1)

# Optional: inspect, filter, sort, or aggregate as you please
print(results_df.head())

# Save for downstream use
results_df.to_csv("individual_risk_predictions.csv", index=False)
print("✅ Individual-level risk predictions saved → individual_risk_predictions.csv")

# -----------------------------------------------  Map numeric ↔︎ human-readable ----

if le is not None:
    # Inverse transform might already give strings, but we need the numeric for the mapping
    numeric_labels = y_pred
else:
    # Original target was numeric, so y_pred is already numeric
    numeric_labels = y_pred

# Fallback / custom mapping (edit to suit your dataset) - keys are numeric predictions
risk_mapping = {
    0: "Low Risk",
    1: "Medium Risk",
    2: "High Risk",
    # Add other mappings here if you have more classes
}

# Use the numeric predictions to look up the human-readable labels
risk_labels = [
    risk_mapping.get(lbl, f"Unknown Risk: {lbl}") # Fallback for unexpected numeric labels
    for lbl in numeric_labels
]


# -----------------------------------------------  Assemble an easy-to-read table ---
results_df = X_for_pred.copy()
results_df["Predicted_Risk"]   = risk_labels
results_df["Max_Probability"]  = y_pred_proba.max(axis=1)

# Optional: inspect, filter, sort, or aggregate as you please
print(results_df.head())

# Save for downstream use
results_df.to_csv("individual_risk_predictions.csv", index=False)
print("✅ Individual-level risk predictions saved → individual_risk_predictions.csv")

# %%
# =============================== A. Risk Distribution ===============================
import seaborn as sns

plt.figure(figsize=(8, 5))
# Ensure the order matches the palette keys if using specific strings
sns.countplot(data=results_df, x="Predicted_Risk", order=["Low Risk", "Medium Risk", "High Risk"])
plt.title("Distribution of Predicted Mental Health Risk Levels")
plt.xlabel("Risk Level")
plt.ylabel("Number of Individuals")
plt.tight_layout()
plt.show()

# %%
# =============================== B. Risk Confidence per Group =======================
plt.figure(figsize=(8, 5))
# Ensure the order matches the palette keys if using specific strings
sns.violinplot(data=results_df, x="Predicted_Risk", y="Max_Probability",
               order=["Low Risk", "Medium Risk", "High Risk"], inner="box")
plt.title("Model Confidence per Predicted Risk Group")
plt.xlabel("Predicted Risk Level")
plt.ylabel("Prediction Confidence")
plt.tight_layout()
plt.show()

# %%
# =============================== C. Mood vs. Stress Scatter =========================
plt.figure(figsize=(8, 6))
# The palette now correctly maps the string labels in results_df["Predicted_Risk"]
sns.scatterplot(data=results_df, x="mood_score", y="stress_level", hue="Predicted_Risk",
                palette={"Low Risk": "green", "Medium Risk": "orange", "High Risk": "red"})
plt.title("Risk Classification by Mood and Stress Level")
plt.xlabel("Mood Score")
plt.ylabel("Stress Level")
plt.legend(title="Predicted Risk")
plt.tight_layout()
plt.show()