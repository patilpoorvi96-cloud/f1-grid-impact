"""Podium-probability model.

We predict whether a driver finishes on the podium (top 3) using:
    - grid position (where they start)
    - prev_pts: the team's average points per entry in the previous season
      (a proxy for how fast the car is)

Only the modern era (2014 onwards) is used, because rules and team
strengths were very different before that.

Finding from testing: adding circuit and team *names* did not help, because
team strength changes every season. Last season's team form did help.
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_prep import load_data

MODERN_ERA_START = 2014
TEST_FROM_YEAR = 2022  # train on earlier seasons, test on later ones
FEATURES_FULL = ["grid", "prev_pts"]
FEATURES_BASE = ["grid"]


def build_pipeline(features) -> Pipeline:
    return Pipeline(
        [
            ("prep", ColumnTransformer([("num", StandardScaler(), list(features))])),
            ("clf", LogisticRegression(max_iter=1000, C=0.3)),
        ]
    )


def modern_data(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["year"] >= MODERN_ERA_START].copy()


def evaluate(df: pd.DataFrame) -> dict:
    """Train on older seasons, test on newer ones (no peeking at the future)."""
    data = modern_data(df)
    train = data[data["year"] < TEST_FROM_YEAR]
    test = data[data["year"] >= TEST_FROM_YEAR]
    if train.empty or test.empty:
        return {}

    results = {}
    for label, feats in [("grid_only", FEATURES_BASE), ("grid_plus_team_form", FEATURES_FULL)]:
        pipe = build_pipeline(feats)
        pipe.fit(train[feats], train["podium"])
        proba = pipe.predict_proba(test[feats])[:, 1]
        results[label] = {
            "accuracy": round(accuracy_score(test["podium"], proba >= 0.5), 3),
            "roc_auc": round(roc_auc_score(test["podium"], proba), 3),
        }
    results["baseline_accuracy_always_no_podium"] = round(1 - test["podium"].mean(), 3)
    results["train_rows"] = len(train)
    results["test_rows"] = len(test)
    return results


def train_final_model(df: pd.DataFrame) -> Pipeline:
    """Fit on all modern-era data for use in the app."""
    data = modern_data(df)
    pipe = build_pipeline(FEATURES_FULL)
    pipe.fit(data[FEATURES_FULL], data["podium"])
    return pipe


def predict_podium_probability(model: Pipeline, grid: int, team_form: float) -> float:
    row = pd.DataFrame([{"grid": grid, "prev_pts": team_form}])
    return float(model.predict_proba(row)[0, 1])


if __name__ == "__main__":
    metrics = evaluate(load_data())
    for key, value in metrics.items():
        print(f"{key}: {value}")
