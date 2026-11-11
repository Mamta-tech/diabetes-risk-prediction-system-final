"""Train a Random Forest model on the public Pima Indians Diabetes dataset."""
from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

DATA_URL = "https://raw.githubusercontent.com/mikeizbicki/datasets/master/csv/uci/pima-indians-diabetes.csv"
COLUMNS = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
FEATURES = COLUMNS[:-1]


def main():
    print("Downloading the public Pima Indians Diabetes dataset...")
    try:
        data = pd.read_csv(DATA_URL, header=None, names=COLUMNS, timeout=60)
    except Exception as exc:
        raise RuntimeError(
            "Could not download the Pima diabetes dataset during the Render build. "
            "Check that the build environment can access GitHub, then redeploy. "
            f"Original error: {exc}"
        ) from exc
    # In this dataset, zero values in these measurements represent missing readings.
    for col in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
        data[col] = data[col].replace(0, pd.NA)
    X, y = data[FEATURES], data["Outcome"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced"))
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f"Test accuracy: {accuracy_score(y_test, predictions):.3f}")
    print(classification_report(y_test, predictions, zero_division=0))
    joblib.dump({"model": model, "features": FEATURES}, Path(__file__).with_name("diabetes_model.pkl"))
    print("Saved diabetes_model.pkl")

if __name__ == "__main__":
    main()
