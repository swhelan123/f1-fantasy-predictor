"""
predict.py

Loads the trained LightGBM model and generates predicted F1 Fantasy points
for every driver and constructor for the next race weekend.

How it works:
  - Identifies the next upcoming race from the F1 schedule
  - Builds a feature row for each driver using their most recent rolling stats
  - Runs predictions through the saved model
  - Outputs a ranked prediction table ready for the optimiser

Usage:
    python src/models/predict.py
    python src/models/predict.py --season 2026 --round 1
"""

import argparse
import logging
import pickle
from pathlib import Path

import duckdb
import fastf1
import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "f1_fantasy.duckdb"
MODEL_PATH = ROOT / "models" / "lgbm_predictor.pkl"
PARQUET = ROOT / "data" / "processed" / "features.parquet"
CACHE_DIR = ROOT / "fastf1_cache"
PROCESSED = ROOT / "data" / "processed"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)

fastf1.Cache.enable_cache(str(CACHE_DIR))

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Load Model ────────────────────────────────────────────────────────────────


def load_model(path: Path = MODEL_PATH) -> tuple:
    """Load pickled model bundle → (model, feature_cols, metrics)."""
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    metrics = bundle["metrics"]
    log.info(
        "Model loaded — trained MAE: %.2f  R²: %.3f", metrics["mae"], metrics["r2"]
    )
    return model, feature_cols, metrics


# ── Identify Next Race ────────────────────────────────────────────────────────


def get_next_race(season: int, round_number: int | None = None) -> dict:
    """
    Return metadata for the next upcoming race.
    If round_number is specified, use that directly.
    """
    schedule = fastf1.get_event_schedule(season, include_testing=False)
    schedule = schedule[schedule["EventFormat"] != "testing"]

    if round_number:
        row = schedule[schedule["RoundNumber"] == round_number].iloc[0]
    else:
        today = pd.Timestamp.utcnow().tz_localize(None)
        future = schedule[schedule["Session5DateUtc"].dt.tz_localize(None) > today]
        if future.empty:
            log.warning("No future races found for %d — using last round", season)
            row = schedule.iloc[-1]
        else:
            row = future.iloc[0]

    race = {
        "Season": season,
        "Round": int(row["RoundNumber"]),
        "EventName": row["EventName"],
        "Location": row["Location"],
        "Country": row["Country"],
        "HasSprint": row["EventFormat"] == "sprint_shootout",
    }
    log.info(
        "Next race: %s R%d — %s (%s)",
        season,
        race["Round"],
        race["EventName"],
        race["Location"],
    )
    return race


# ── Build Prediction Features ─────────────────────────────────────────────────


def build_prediction_features(
    race: dict,
    history: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build one feature row per driver for the upcoming race.
    Uses rolling stats from the most recent 3 completed races.

    history: the full features.parquet dataframe (all past races)
    """
    log.info("Building prediction features for %s...", race["EventName"])

    # Get the most recent data per driver (their current rolling state)
    latest = (
        history.sort_values(["Season", "Round"]).groupby("Driver").last().reset_index()
    )

    rows = []
    for _, driver_row in latest.iterrows():
        driver = driver_row["Driver"]

        # Circuit-specific history for this location
        circuit_hist = history[
            (history["Driver"] == driver) & (history["Location"] == race["Location"])
        ]

        row = {
            # Identity
            "Season": race["Season"],
            "Round": race["Round"],
            "Driver": driver,
            "Constructor": driver_row["Constructor"],
            "EventName": race["EventName"],
            "Location": race["Location"],
            # Rolling form — carry forward from last race
            "RollingAvgFinish": driver_row.get("RollingAvgFinish", np.nan),
            "RollingAvgFantasyPts": driver_row.get("RollingAvgFantasyPts", np.nan),
            "RollingDNFRate": driver_row.get("RollingDNFRate", np.nan),
            "RollingAvgPositionsGained": driver_row.get(
                "RollingAvgPositionsGained", np.nan
            ),
            "RollingAvgQualiPos": driver_row.get("RollingAvgQualiPos", np.nan),
            "RollingPPM_Proxy": driver_row.get("RollingPPM_Proxy", np.nan),
            # Circuit history (NaN if first visit)
            "CircuitAvgFinish": circuit_hist["RacePosition"].mean()
            if len(circuit_hist)
            else np.nan,
            "CircuitDNFRate": circuit_hist["DNF"].astype(float).mean()
            if len(circuit_hist)
            else np.nan,
            "CircuitAvgFantasyPts": circuit_hist["TotalFantasyPts"].mean()
            if len(circuit_hist)
            else np.nan,
            # Teammate deltas (carry forward rolling average)
            "QualiVsTeammate": driver_row.get("QualiVsTeammate", np.nan),
            "RaceVsTeammate": driver_row.get("RaceVsTeammate", np.nan),
            "AvgTeammateFantasyPts": driver_row.get("AvgTeammateFantasyPts", np.nan),
            # Pit stop features (constructor level — filled below)
            "RollingAvgPitStop": driver_row.get("RollingAvgPitStop", np.nan),
            "RollingPitStopStd": driver_row.get("RollingPitStopStd", np.nan),
            # Weather — unknown until race week, use circuit historical avg
            "AvgAirTemp": history[history["Location"] == race["Location"]][
                "AvgAirTemp"
            ].mean(),
            "AvgTrackTemp": history[history["Location"] == race["Location"]][
                "AvgTrackTemp"
            ].mean(),
            "AvgHumidity": history[history["Location"] == race["Location"]][
                "AvgHumidity"
            ].mean(),
            "AvgWindSpeed": history[history["Location"] == race["Location"]][
                "AvgWindSpeed"
            ].mean(),
            "Rainfall": int(
                history[history["Location"] == race["Location"]]["Rainfall"].mean()
                > 0.3
            ),
            # Sprint flag
            "HasSprint": int(race["HasSprint"]),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Join pre-season testing features
    try:
        con = duckdb.connect(str(DB_PATH))
        testing = con.execute("SELECT * FROM testing_results").df()
        con.close()
        testing_cols = [
            "Season",
            "Driver",
            "TestFastestLap",
            "TestLongRunPace",
            "TestTotalLaps",
            "TestFastestLapRank",
            "TestLongRunRank",
        ]
        testing = testing[testing_cols]
        df = df.merge(testing, on=["Season", "Driver"], how="left")
        matched = df["TestFastestLap"].notna().sum()
        log.info("Testing features joined: %d/%d drivers matched", matched, len(df))
    except Exception as e:
        log.warning("Could not join testing features: %s", e)
        for col in [
            "TestFastestLap",
            "TestLongRunPace",
            "TestTotalLaps",
            "TestFastestLapRank",
            "TestLongRunRank",
        ]:
            df[col] = np.nan

    # Re-encode categoricals to match training encoding
    history_enc = history.copy()
    for col in ["Driver", "Constructor", "Location"]:
        cat = history_enc[col].astype("category")
        df[col + "_enc"] = (
            df[col]
            .map(dict(zip(cat.cat.categories, range(len(cat.cat.categories)))))
            .fillna(-1)
            .astype(int)
        )

    log.info("Built %d driver prediction rows", len(df))
    return df


# ── Run Predictions ───────────────────────────────────────────────────────────


def predict(
    pred_df: pd.DataFrame,
    model,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Run the model on the prediction feature rows.
    Returns pred_df with PredictedPts and PredictedRank columns added.
    """
    available = [f for f in feature_cols if f in pred_df.columns]
    X = pred_df[available].copy()

    # Fill NaN with column medians from training features
    history = pd.read_parquet(PARQUET)
    for col in X.columns:
        if X[col].isna().any():
            fill_val = history[col].median() if col in history.columns else 0
            X[col] = X[col].fillna(fill_val)

    pred_df = pred_df.copy()
    pred_df["PredictedPts"] = model.predict(X)
    pred_df["PredictedRank"] = pred_df["PredictedPts"].rank(ascending=False).astype(int)

    return pred_df.sort_values("PredictedPts", ascending=False).reset_index(drop=True)


# ── Aggregate Constructor Predictions ────────────────────────────────────────


def aggregate_constructors(pred_df: pd.DataFrame) -> pd.DataFrame:
    """
    Sum the top 2 driver predictions per constructor to get
    constructor predicted points (mirrors how F1 Fantasy scores constructors).
    """
    ctor = (
        pred_df.groupby("Constructor")
        .apply(lambda g: g.nlargest(2, "PredictedPts")["PredictedPts"].sum())
        .reset_index()
        .rename(columns={0: "PredictedPts"})
        .sort_values("PredictedPts", ascending=False)
        .reset_index(drop=True)
    )
    ctor["PredictedRank"] = range(1, len(ctor) + 1)
    return ctor


# ── Display & Save ────────────────────────────────────────────────────────────


def display_predictions(
    driver_preds: pd.DataFrame,
    ctor_preds: pd.DataFrame,
    race: dict,
):
    """Log a clean summary of predictions."""
    log.info(
        "━━━ Predictions: %s R%d — %s ━━━",
        race["Season"],
        race["Round"],
        race["EventName"],
    )

    log.info("── Top 10 Drivers ──")
    top_drivers = driver_preds[["Driver", "Constructor", "PredictedPts"]].head(10)
    top_drivers = top_drivers.copy()
    top_drivers["PredictedPts"] = top_drivers["PredictedPts"].round(1)
    log.info("\n%s", top_drivers.to_string(index=False))

    log.info("── Constructors ──")
    ctor_display = ctor_preds[["Constructor", "PredictedPts"]].copy()
    ctor_display["PredictedPts"] = ctor_display["PredictedPts"].round(1)
    log.info("\n%s", ctor_display.to_string(index=False))


def save_predictions(
    driver_preds: pd.DataFrame,
    ctor_preds: pd.DataFrame,
    race: dict,
):
    """Save predictions to parquet for the optimiser to consume."""
    driver_path = (
        PROCESSED
        / f"predictions_drivers_S{race['Season']}_R{race['Round']:02d}.parquet"
    )
    ctor_path = (
        PROCESSED
        / f"predictions_constructors_S{race['Season']}_R{race['Round']:02d}.parquet"
    )

    driver_preds.to_parquet(driver_path, index=False)
    ctor_preds.to_parquet(ctor_path, index=False)

    # Also save as "latest" for the optimiser to always find
    driver_preds.to_parquet(
        PROCESSED / "predictions_drivers_latest.parquet", index=False
    )
    ctor_preds.to_parquet(
        PROCESSED / "predictions_constructors_latest.parquet", index=False
    )

    log.info("Predictions saved → %s", driver_path.name)
    log.info("Predictions saved → %s", ctor_path.name)


# ── Main ──────────────────────────────────────────────────────────────────────


def run(
    season: int = 2026, round_number: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    log.info("━━━ F1 Fantasy — Prediction Pipeline ━━━")

    model, feature_cols, metrics = load_model()
    history = pd.read_parquet(PARQUET)
    race = get_next_race(season, round_number)

    pred_df = build_prediction_features(race, history)
    driver_preds = predict(pred_df, model, feature_cols)
    ctor_preds = aggregate_constructors(driver_preds)

    display_predictions(driver_preds, ctor_preds, race)
    save_predictions(driver_preds, ctor_preds, race)

    log.info("━━━ Done ━━━")
    return driver_preds, ctor_preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate F1 Fantasy predictions")
    parser.add_argument("--season", type=int, default=2026, help="F1 season year")
    parser.add_argument("--round", type=int, default=None, help="Specific round number")
    args = parser.parse_args()

    run(season=args.season, round_number=args.round)
