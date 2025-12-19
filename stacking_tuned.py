import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.feature_selection import SelectFromModel
from scipy.stats import randint, uniform

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
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric_features)
    ])

# HGB (gradient boosting method)
hgb = HistGradientBoostingClassifier(random_state=42)

# Random Forest (bagging method)
rf = RandomForestClassifier(
    class_weight="balanced", 
    random_state=42,
    n_jobs=-1
)

# Extra Trees (adding randomness to avoid overfitting)
et = ExtraTreesClassifier(
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
    ("selector", SelectFromModel(RandomForestClassifier(n_estimators=50, random_state=42), threshold="median")),
    ("stack", stacking_clf)
])

# hyperparameter search space
param_distributions = {
    # HistGradientBoostingClassifier parameters
    "stack__hgb__learning_rate": uniform(0.01, 0.19),  # 0.01 to 0.2
    "stack__hgb__max_iter": randint(100, 500),
    "stack__hgb__max_depth": randint(3, 10),
    "stack__hgb__min_samples_leaf": randint(10, 50),
    "stack__hgb__l2_regularization": uniform(0, 1),
    
    # RandomForest parameters
    "stack__rf__n_estimators": randint(100, 400),
    "stack__rf__max_depth": [None, 10, 20, 30],
    "stack__rf__min_samples_split": randint(2, 20),
    "stack__rf__min_samples_leaf": randint(1, 10),
    "stack__rf__max_features": ["sqrt", "log2", 0.3, 0.5],
    
    # ExtraTrees parameters
    "stack__et__n_estimators": randint(100, 400),
    "stack__et__max_depth": [None, 10, 20, 30],
    "stack__et__min_samples_split": randint(2, 20),
    "stack__et__min_samples_leaf": randint(1, 10),
    "stack__et__max_features": ["sqrt", "log2", 0.3, 0.5],
    
    # Logistic Regression (meta-learner) parameters
    "stack__final_estimator__C": uniform(0.01, 10),
    "stack__final_estimator__penalty": ["l1", "l2"],
    "stack__final_estimator__solver": ["saga"],
    
    # Feature selector threshold
    "selector__threshold": ["median", "mean"]
}

# cross-validation strategy
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# randomized search for hyperparameter tuning
random_search = RandomizedSearchCV(
    final_pipeline,
    param_distributions=param_distributions,
    n_iter=50,  # number of parameter combinations to try
    cv=skf,
    scoring="accuracy",
    n_jobs=-1,
    verbose=2,
    random_state=42,
    return_train_score=True
)

print("Starting hyperparameter tuning...")
random_search.fit(X, y)

print(f"\nBest parameters: {random_search.best_params_}")
print(f"Best cross-validation accuracy: {random_search.best_score_:.4f}")

# evaluate with cross-validation using best estimator
best_pipeline = random_search.best_estimator_
scores = cross_val_score(best_pipeline, X, y, cv=skf, scoring="accuracy")

print(f"\nFinal Mean Accuracy: {scores.mean():.4f}")
print(f"Std Dev: {scores.std():.4f}")

# generate predictions with best model
predictions = best_pipeline.predict(X_test)
probs = best_pipeline.predict_proba(X_test)

submission = pd.DataFrame({
    "id": test_df["id"],
    "class4": predictions,
    "p": np.max(probs, axis=1)
})
submission.to_csv("submission_stacking.csv", index=False)

# save results summary
results_df = pd.DataFrame(random_search.cv_results_)
results_df.to_csv("hyperparameter_tuning_results.csv", index=False)
print("\nHyperparameter tuning results saved to hyperparameter_tuning_results.csv")