# -Predictive-Analytics-for-Indian-Crop-Recommendation
import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

warnings.filterwarnings("ignore")

DATA_FILE = "indiancrop_dataset.csv"
OUTPUT_DIR = "crop_project_outputs"
TEST_SIZE = 0.20
RANDOM_STATE = 42
os.makedirs(OUTPUT_DIR, exist_ok=True)

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        f"Could not find '{DATA_FILE}'. Put the CSV file in the same folder as this script "
        "or change DATA_FILE to the full path of your dataset."
    )

data = pd.read_csv(DATA_FILE)
data.columns = data.columns.str.strip()

required_columns = [
    "N_SOIL", "P_SOIL", "K_SOIL", "TEMPERATURE",
    "HUMIDITY", "ph", "RAINFALL", "STATE", "CROP", "CROP_PRICE"
]
missing_columns = [c for c in required_columns if c not in data.columns]
if missing_columns:
    raise ValueError(
        "Missing expected columns: " + ", ".join(missing_columns) +
        "\nColumns found: " + ", ".join(map(str, data.columns))
    )

print("=" * 60)
print("PREDICTIVE ANALYTICS FOR INDIAN CROP RECOMMENDATION")
print("=" * 60)
print("\nDataset shape:", data.shape)
print("\nFirst five rows:\n", data.head())
print("\nDataset information:")
data.info()
print("\nDescriptive statistics:\n", data.describe(include="all").transpose())
print("\nMissing values:\n", data.isnull().sum())
print("\nDuplicate rows:", data.duplicated().sum())
print("\nCrop counts:\n", data["CROP"].value_counts().sort_index())

data.describe(include="all").transpose().to_csv(os.path.join(OUTPUT_DIR, "dataset_summary.csv"))
data.isnull().sum().rename("missing_values").to_csv(os.path.join(OUTPUT_DIR, "missing_values.csv"))
data["CROP"].value_counts().sort_index().rename("count").to_csv(os.path.join(OUTPUT_DIR, "crop_counts.csv"))

numeric_columns = ["N_SOIL", "P_SOIL", "K_SOIL", "TEMPERATURE", "HUMIDITY", "ph", "RAINFALL", "CROP_PRICE"]
eda_features = ["N_SOIL", "P_SOIL", "K_SOIL", "TEMPERATURE", "HUMIDITY", "ph", "RAINFALL"]

plt.figure(figsize=(12, 6))
crop_counts = data["CROP"].value_counts().sort_values(ascending=False)
plt.bar(crop_counts.index.astype(str), crop_counts.values)
plt.title("Distribution of Crop Classes")
plt.xlabel("Crop")
plt.ylabel("Number of Records")
plt.xticks(rotation=75, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "crop_distribution.png"), dpi=150)
plt.show()
plt.close()

data[numeric_columns].hist(figsize=(14, 10), bins=20)
plt.suptitle("Distribution of Numerical Features")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "numeric_distributions.png"), dpi=150)
plt.show()
plt.close("all")

corr = data[numeric_columns].corr(numeric_only=True)
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
fig.colorbar(im, ax=ax, label="Pearson correlation")
ax.set_xticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=60, ha="right")
ax.set_yticks(range(len(corr.index)))
ax.set_yticklabels(corr.index)
for i in range(corr.shape[0]):
    for j in range(corr.shape[1]):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
ax.set_title("Correlation Heatmap")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"), dpi=150)
plt.show()
plt.close(fig)

crop_labels = sorted(data["CROP"].dropna().unique(), key=lambda v: str(v))
for feature in eda_features:
    fig, ax = plt.subplots(figsize=(13, 6))
    values = [data.loc[data["CROP"] == crop, feature].dropna().values for crop in crop_labels]
    ax.boxplot(values, labels=[str(crop) for crop in crop_labels], showfliers=True)
    ax.set_title(f"{feature} Across Crop Classes")
    ax.set_xlabel("Crop")
    ax.set_ylabel(feature)
    ax.tick_params(axis="x", labelrotation=75, labelsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"boxplot_{feature.lower()}.png"), dpi=150)
    plt.show()
    plt.close(fig)

state_crop = pd.crosstab(data["STATE"], data["CROP"])
fig, ax = plt.subplots(figsize=(14, 8))
im = ax.imshow(state_crop.values, aspect="auto", cmap="YlGnBu")
fig.colorbar(im, ax=ax, label="Record count")
ax.set_xticks(range(len(state_crop.columns)))
ax.set_xticklabels(state_crop.columns.astype(str), rotation=90, fontsize=7)
ax.set_yticks(range(len(state_crop.index)))
ax.set_yticklabels(state_crop.index.astype(str), fontsize=8)
ax.set_title("Crop Records Across States")
ax.set_xlabel("Crop")
ax.set_ylabel("State")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "state_crop_heatmap.png"), dpi=150)
plt.show()
plt.close(fig)
state_crop.to_csv(os.path.join(OUTPUT_DIR, "state_crop_counts.csv"))

X = data.drop(columns=["CROP", "CROP_PRICE"]).copy()
y = data["CROP"].copy()

numeric_features = ["N_SOIL", "P_SOIL", "K_SOIL", "TEMPERATURE", "HUMIDITY", "ph", "RAINFALL"]
categorical_features = ["STATE"]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

stratify_target = y if y.value_counts().min() >= 2 else None
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify_target
)

models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "SVM": SVC(kernel="rbf"),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
}

results = []
labels = sorted(y.unique(), key=lambda v: str(v))

for model_name, classifier in models.items():
    print("\n" + "=" * 60 + "\n" + model_name + "\n" + "=" * 60)
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Weighted Precision": precision,
        "Weighted Recall": recall,
        "Weighted F1 Score": f1
    })

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Weighted Precision: {precision:.4f}")
    print(f"Weighted Recall: {recall:.4f}")
    print(f"Weighted F1 Score: {f1:.4f}")
    print("\nClassification report:\n", classification_report(y_test, y_pred, zero_division=0))

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    pd.DataFrame(
        cm,
        index=[f"Actual {label}" for label in labels],
        columns=[f"Predicted {label}" for label in labels]
    ).to_csv(os.path.join(OUTPUT_DIR, f"{model_name.lower().replace(' ', '_')}_confusion_matrix.csv"))

    fig, ax = plt.subplots(figsize=(12, 9))
    im = ax.imshow(cm, aspect="auto", cmap="Blues")
    fig.colorbar(im, ax=ax, label="Number of records")
    ax.set_title(f"Confusion Matrix - {model_name}")
    ax.set_xlabel("Predicted Crop")
    ax.set_ylabel("Actual Crop")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([str(label) for label in labels], rotation=90, fontsize=7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels([str(label) for label in labels], fontsize=7)
    if len(labels) <= 25:
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if cm[i, j] != 0:
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png"), dpi=150)
    plt.show()
    plt.close(fig)

results_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False)
results_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)
print("\n" + "=" * 60 + "\nFINAL MODEL COMPARISON\n" + "=" * 60)
print(results_df.round(4).to_string(index=False))
print("\nOutput files saved in:", os.path.abspath(OUTPUT_DIR))
