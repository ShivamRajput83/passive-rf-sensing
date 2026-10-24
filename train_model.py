"""Train and evaluate a Random Forest on the synthetic feature dataset."""
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

DATA_PATH = "dataset.csv"
FEATURES = [
    "mean_power", "peak_power", "power_variance",
    "spectral_entropy", "spectral_peak_fraction"
]

df = pd.read_csv(DATA_PATH)
X = df[FEATURES]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=200, random_state=42, class_weight="balanced"
)
model.fit(X_train, y_train)
pred = model.predict(X_test)

print(f"Test accuracy on SIMULATED data: {accuracy_score(y_test, pred):.3f}")
print("\nClassification report:")
print(classification_report(y_test, pred))
print("Confusion matrix:")
print(confusion_matrix(y_test, pred, labels=model.classes_))

joblib.dump({"model": model, "features": FEATURES}, "rf_simulated_model.joblib")
print("\nSaved rf_simulated_model.joblib")
print("Do not report this synthetic-data accuracy as real smartphone detection accuracy.")
