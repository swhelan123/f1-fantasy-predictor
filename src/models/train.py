"""
train.py

Trains a LightGBM regression model to predict F1 Fantasy points per driver
per race weekend. Uses the feature table built by engineer.py.

Key design decisions:
  - Time-based train/validation split (never shuffle F1 data — it's temporal)
  - Optuna hyperparameter tuning
  - SHAP values logged for interpretability
  - MLflow experiment tracking
  - Model saved to models/lgbm_predictor.pkl

Usage:
    python src/models/train.py
    python src/models/train.py --tune      # run Optuna HPO (slower)
    python src/models/train.py --seasons 2025  # train on single season
"""

import argparse
import logging
import pickle
import warnings
from pathlib import Path

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import optuna
import pandas as pd
import shap
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "f1_fantasy.duckdb"
PARQUET = ROOT / "data" / "processed" / "features.parquet"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODELS_DIR / "lgbm_predictor.pkl"

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Feature / Target Config ───────────────────────────────────────────────────

TARGET = "TotalFantasyPts"

# Features fed to LightGBM — excludes identity cols and the target itself
ML_FEATURES = [
    # Rolling form
    "RollingAvgFinish",
    "RollingAvgFantasyPts",
    "RollingDNFRate",
    "RollingAvgPositionsGained",
    "RollingAvgQualiPos",
    # Circuit history
    "CircuitAvgFinish",
    "CircuitDNFRate",
    "CircuitAvgFantasyPts",
    # Teammate deltas
    "QualiVsTeammate",
    "RaceVsTeammate",
    "AvgTeammateFantasyPts",
    # Pit stops
    "RollingAvgPitStop",
    "RollingPitStopStd",
    # Weather
    "AvgAirTemp",
    "AvgTrackTemp",
    "AvgHumidity",
    "AvgWindSpeed",
    "Rainfall",
    # PPM proxy
    "RollingPPM_Proxy",
    # Encoded categoricals
    "Driver_enc",
    "Constructor_enc",
    "Location_enc",
]

# LightGBM categorical feature indices (will be set at runtime)
CATEGORICAL_FEATURES = ["Driver_enc", "Constructor_enc", "Location_enc"]

# Default hyperparameters (used when --tune is not passed)
DEFAULT_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "n_estimators": 500,
    "verbose": -1,
    "random_state": 42,
}


# ── Data Loading ──────────────────────────────────────────────────────────────


def load_features(seasons: list[int] | None = None) -> pd.DataFrame:
    """Load feature parquet, optionally filtering to specific seasons."""
    df = pd.read_parquet(PARQUET)
    if seasons:
        df = df[df["Season"].isin(seasons)]
    log.info("Loaded %d rows from features.parquet", len(df))
    return df


# ── Time-Based Train/Val Split ────────────────────────────────────────────────


def time_split(
    df: pd.DataFrame, val_rounds: int = 4
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data temporally — last N rounds of the most recent season = validation.
    Never shuffle F1 data; future rounds must not appear in training.
    """
    df = df.sort_values(["Season", "Round"]).reset_index(drop=True)

    # Validation = last val_rounds rounds of the most recent season
    max_season = df["Season"].max()
    season_df = df[df["Season"] == max_season]
    max_round = season_df["Round"].max()
    val_start = max_round - val_rounds + 1

    val_mask = (df["Season"] == max_season) & (df["Round"] >= val_start)
    train_mask = ~val_mask

    train = df[train_mask].copy()
    val = df[val_mask].copy()

    log.info(
        "Train: %d rows (%d seasons, rounds up to %s R%d)",
        len(train),
        train["Season"].nunique(),
        max_season,
        val_start - 1,
    )
    log.info(
        "Val:   %d rows (S%d rounds %d–%d)", len(val), max_season, val_start, max_round
    )

    return train, val


# ── Prepare X/y ──────────────────────────────────────────────────────────────


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return feature matrix X and target y, filling NaNs."""
    available = [f for f in ML_FEATURES if f in df.columns]
    X = df[available].copy()

    # Fill NaN with median per column (circuit history NaN on first visit etc.)
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())

    y = df[TARGET].copy()
    return X, y


# ── Optuna HPO ────────────────────────────────────────────────────────────────


def tune_hyperparams(X_train, y_train, X_val, y_val, n_trials: int = 50) -> dict:
    """Run Optuna to find best LightGBM hyperparameters."""
    log.info("Running Optuna HPO (%d trials)...", n_trials)

    def objective(trial):
        params = {
            "objective": "regression",
            "metric": "mae",
            "boosting_type": "gbdt",
            "verbose": -1,
            "random_state": 42,
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "n_estimators": 500,
        }
        model = lgb.LGBMRegressor(**params)
        cat_feats = [f for f in CATEGORICAL_FEATURES if f in X_train.columns]
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            categorical_feature=cat_feats,
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )
        preds = model.predict(X_val)
        return mean_absolute_error(y_val, preds)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    best.update(
        {
            "objective": "regression",
            "metric": "mae",
            "boosting_type": "gbdt",
            "verbose": -1,
            "random_state": 42,
            "n_estimators": 500,
        }
    )
    log.info("Best MAE: %.3f | params: %s", study.best_value, best)
    return best


# ── Train ─────────────────────────────────────────────────────────────────────


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: dict,
) -> lgb.LGBMRegressor:
    """Train LightGBM with early stopping on validation set."""
    log.info("Training LightGBM...")
    cat_feats = [f for f in CATEGORICAL_FEATURES if f in X_train.columns]

    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        categorical_feature=cat_feats,
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(50),
        ],
    )
    log.info("Best iteration: %d", model.best_iteration_)
    return model


# ── Evaluate ──────────────────────────────────────────────────────────────────


def evaluate(
    model, X_val: pd.DataFrame, y_val: pd.Series, val_df: pd.DataFrame
) -> dict:
    """Compute validation metrics and log top/bottom predictions."""
    preds = model.predict(X_val)

    mae = mean_absolute_error(y_val, preds)
    rmse = mean_squared_error(y_val, preds) ** 0.5
    r2 = r2_score(y_val, preds)

    log.info("── Validation Metrics ──")
    log.info("  MAE:  %.2f pts", mae)
    log.info("  RMSE: %.2f pts", rmse)
    log.info("  R²:   %.3f", r2)

    # Show best predicted drivers vs actual
    results = val_df[["Season", "Round", "Driver", "Constructor", "EventName"]].copy()
    results["Actual"] = y_val.values
    results["Predicted"] = preds
    results["Error"] = preds - y_val.values

    log.info("── Top 5 Predictions (by predicted pts) ──")
    top = results.nlargest(5, "Predicted")[
        ["Driver", "EventName", "Actual", "Predicted", "Error"]
    ]
    log.info("\n%s", top.to_string(index=False))

    return {"mae": mae, "rmse": rmse, "r2": r2, "predictions": results}


# ── SHAP Feature Importance ───────────────────────────────────────────────────


def log_shap(model, X_val: pd.DataFrame):
    """Compute and log SHAP feature importance."""
    log.info("Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_val)
    importance = pd.DataFrame(
        {
            "feature": X_val.columns,
            "shap_mean": np.abs(shap_vals).mean(axis=0),
        }
    ).sort_values("shap_mean", ascending=False)

    log.info("── Top 10 Features by SHAP ──")
    log.info("\n%s", importance.head(10).to_string(index=False))
    return importance


# ── Save Model ────────────────────────────────────────────────────────────────


def save_model(model, feature_cols: list[str], metrics: dict):
    """Pickle model + metadata for use by predict.py."""
    bundle = {
        "model": model,
        "feature_cols": feature_cols,
        "metrics": metrics,
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)
    log.info("Model saved to %s", MODEL_PATH)


# ── Main ──────────────────────────────────────────────────────────────────────


def run(tune: bool = False, seasons: list[int] | None = None, val_rounds: int = 4):
    log.info("━━━ F1 Fantasy — LightGBM Training ━━━")

    df = load_features(seasons)
    train_df, val_df = time_split(df, val_rounds=val_rounds)

    X_train, y_train = prepare_xy(train_df)
    X_val, y_val = prepare_xy(val_df)

    log.info(
        "Features: %d | Train rows: %d | Val rows: %d",
        len(X_train.columns),
        len(X_train),
        len(X_val),
    )

    # Hyperparameter tuning (optional)
    params = (
        tune_hyperparams(X_train, y_train, X_val, y_val) if tune else DEFAULT_PARAMS
    )
    log.info("Using params: %s", params)

    # MLflow tracking
    mlflow.set_experiment("f1_fantasy_predictor")
    with mlflow.start_run():
        mlflow.log_params(params)

        model = train_model(X_train, y_train, X_val, y_val, params)
        metrics = evaluate(model, X_val, y_val, val_df)
        shap_df = log_shap(model, X_val)

        mlflow.log_metrics(
            {
                "val_mae": metrics["mae"],
                "val_rmse": metrics["rmse"],
                "val_r2": metrics["r2"],
            }
        )
        mlflow.lightgbm.log_model(model, "lgbm_model")

    save_model(model, list(X_train.columns), metrics)

    log.info("━━━ Done ━━━")
    return model, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train F1 Fantasy LightGBM model")
    parser.add_argument("--tune", action="store_true", help="Run Optuna HPO")
    parser.add_argument(
        "--seasons", nargs="+", type=int, help="Seasons to train on e.g. 2024 2025"
    )
    parser.add_argument(
        "--val-rounds", type=int, default=4, help="Rounds to hold out for validation"
    )
    args = parser.parse_args()

    run(
        tune=args.tune,
        seasons=args.seasons,
        val_rounds=args.val_rounds,
    )
