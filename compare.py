import yaml
import numpy as np
import pandas as pd
from typing import Tuple
from pathlib import Path
from itertools import product
from xgboost import XGBClassifier
from sklearn.svm import LinearSVC
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)

import numpy as np
np.random.seed(42)

CSV_NAME = "data_clean.csv"
TARGET_COL = "SuccessfulBool"
COLS_YAML = "cols.yaml"

def split_X_y(df: pd.DataFrame, y_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Return numeric X (inputs) and y (output)"""
    y: pd.Series = df[y_col].copy() # type: ignore
    X = df.drop(columns=[y_col]).copy()
    return X, y

def main():
    csv_path = Path(CSV_NAME)
    target_col = TARGET_COL
    with open(COLS_YAML, "r") as f:
        config = yaml.safe_load(f)
    numeric_cols = config["numeric_cols"]
    categorical_cols = config["categorical_cols"]
    text_cols = config["text_cols"]

    # Load data and clean data
    print(f"Loading data from {csv_path.resolve()}")
    df = pd.read_csv(csv_path)

    # Feature engineering
    df["text"] = df[text_cols].agg(" ".join, axis=1)
    df.drop(columns=text_cols, inplace=True)

    # Split X and y
    y_name = target_col
    X, y = split_X_y(df, y_name)
    for col in categorical_cols:
        X[col] = X[col].astype("category")

    logreg_prep = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("txt", TfidfVectorizer(
                max_features=30000,
                ngram_range=(1, 2),
                min_df=5
            ), "text"),
        ]
    )
    
    models = {}
    catboost_params = {
        "depth": [9],
        "l2_leaf_reg": [15],
        "border_count": [16],
    }
    for depth, l2, bc in product(
        catboost_params["depth"],
        catboost_params["l2_leaf_reg"],
        catboost_params["border_count"],
    ):
        name = f"Cat_d{depth}_l2{l2}_b{bc}"

        models[name] = CatBoostClassifier(
            text_features=["text"],
            cat_features=categorical_cols,
            n_estimators=300,
            learning_rate=0.1,
            depth=depth,
            l2_leaf_reg=l2,
            border_count=bc,
            random_state=42,
            verbose=False,
        )
    #     models[name].fit(X, y)
    #     print(models[name].get_feature_importance(prettified=True))
    logreg_param_grid = {
        "penalty": ["l2"],
        "C": [0.5, 1.0, 2.0, 4.0],
    }
    for pen, C in product(logreg_param_grid["penalty"], logreg_param_grid["C"]):
        name = f"LogReg_{pen}_C{C}"
        clf = LogisticRegression(
            penalty=pen,
            solver="saga",
            C=C,
            max_iter=5000,
            random_state=42,
            n_jobs=-1,
        )
        models[name] = Pipeline([
            ("prep", logreg_prep),
            ("clf", clf),
        ])

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print("\nCross-validation:\n")
    for name, model in models.items():
        cv_results = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring="accuracy",
            return_train_score=True,
            n_jobs=-1,
        )
        train_mean = cv_results["train_score"].mean()
        train_std  = cv_results["train_score"].std()
        val_mean   = cv_results["test_score"].mean()
        val_std    = cv_results["test_score"].std()

        print(f"{name:20s} | "
              f"train={train_mean:.4f}±{train_std:.4f}  "
              f"val={val_mean:.4f}±{val_std:.4f}")

if __name__ == "__main__":
    main()
