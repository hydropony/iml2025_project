import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_selection import SelectFromModel

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# data handling
def get_date_features(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"]/12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"]/12)
    return df.drop(["date", "id"], axis=1)

X = get_date_features(train_df.drop("class4", axis=1))
y = train_df["class4"]
X_test = get_date_features(test_df)

if "partlybad" in X.columns:
    X["partlybad"] = X["partlybad"].astype(int)
    X_test["partlybad"] = X_test["partlybad"].astype(int)

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns

preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")), # fill in missing values
            ("scaler", StandardScaler())
        ]), numeric_features)
    ])

# HGB (gradient boosting method)
hgb = HistGradientBoostingClassifier(
    learning_rate=0.05, 
    max_iter=500, 
    max_depth=6, 
    random_state=42
)

# Random Forest (bagging method)
rf = RandomForestClassifier(
    n_estimators=300, 
    class_weight="balanced", 
    random_state=42,
    n_jobs=-1
)

# Extra Trees (adding randomness to avoid overfitting)
et = ExtraTreesClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# use the predictions of the 3 above to make the final decision
estimators = [
    ("hgb", hgb),
    ("rf", rf),
    ("et", et)
]

stacking_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(), 
    n_jobs=-1,
    cv=5
)

final_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    # to evaluate which columns are important, and reduce noise:
    ("selector", SelectFromModel(RandomForestClassifier(n_estimators=50, random_state=42), threshold="median")),
    ("stack", stacking_clf)
])

# cross-validation (test the accuracy)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(final_pipeline, X, y, cv=skf, scoring="accuracy")

print(f"Mean Accuracy: {scores.mean():.4f}")
print(f"Std Dev: {scores.std():.4f}")

final_pipeline.fit(X, y)
predictions = final_pipeline.predict(X_test)
probs = final_pipeline.predict_proba(X_test)

submission = pd.DataFrame({
    "id": test_df["id"],
    "class4": predictions,
    "p": np.max(probs, axis=1)
})
submission.to_csv("submission_stacking.csv", index=False)
