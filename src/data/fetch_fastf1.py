"""
fetch_fastf1.py

Fetches race results, qualifying results, sprint results, and pit stop data
for the last N race weekends using the FastF1 library and persists everything
to a local DuckDB database for downstream feature engineering.

Usage:
    python src/data/fetch_fastf1.py              # fetches last 3 races (default)
    python src/data/fetch_fastf1.py --races 5    # fetches last 5 races
    python src/data/fetch_fastf1.py --full       # fetches entire current season
"""

import argparse
import logging
import os
from pathlib import Path

import duckdb
import fastf1
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "fastf1_cache"
DB_PATH = ROOT / "data" / "f1_fantasy.duckdb"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
(ROOT / "data" / "raw").mkdir(parents=True, exist_ok=True)
(ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── FastF1 cache ─────────────────────────────────────────────────────────────

fastf1.Cache.enable_cache(str(CACHE_DIR))


# ── Helpers ───────────────────────────────────────────────────────────────────


def get_completed_rounds(season: int) -> list[dict]:
    """Return all completed race rounds for the given season."""
    schedule = fastf1.get_event_schedule(season, include_testing=False)
    completed = schedule[schedule["EventFormat"] != "testing"].copy()

    # Keep only rounds that have already happened
    today = pd.Timestamp.utcnow().tz_localize(None)
    completed = completed[completed["Session5DateUtc"].dt.tz_localize(None) < today]

    return completed[["RoundNumber", "EventName", "Location", "Country"]].to_dict(
        "records"
    )


def fetch_session(
    season: int, round_number: int, session_type: str
) -> pd.DataFrame | None:
    """
    Load a single session (Race / Qualifying / Sprint) and return
    a tidy results DataFrame, or None if the session didn't happen.
    """
    try:
        session = fastf1.get_session(season, round_number, session_type)
        session.load(laps=True, telemetry=False, weather=True, messages=False)
    except Exception as exc:
        log.warning(
            "Could not load %s round %d %s: %s", season, round_number, session_type, exc
        )
        return None

    results = session.results
    if results is None or results.empty:
        return None

    df = results[
        [
            "DriverNumber",
            "Abbreviation",
            "FullName",
            "TeamName",
            "GridPosition",
            "Position",
            "Status",
            "Points",
            "Time",
        ]
    ].copy()

    df.rename(
        columns={"Abbreviation": "Driver", "TeamName": "Constructor"}, inplace=True
    )
    df["Season"] = season
    df["Round"] = round_number
    df["SessionType"] = session_type
    df["EventName"] = session.event["EventName"]
    df["Location"] = session.event["Location"]

    # Add fastest lap flag
    if "FastestLap" in results.columns:
        df["FastestLap"] = results["FastestLap"].fillna(False)
    else:
        df["FastestLap"] = False

    # Positions gained/lost (race only)
    if session_type == "Race":
        df["GridPosition"] = pd.to_numeric(df["GridPosition"], errors="coerce")
        df["Position"] = pd.to_numeric(df["Position"], errors="coerce")
        df["PositionsGained"] = df["GridPosition"] - df["Position"]
    else:
        df["PositionsGained"] = None

    # Clean DNF flag
    df["DNF"] = ~df["Status"].str.contains("Finished|Lap", na=False)

    log.info(
        "  ✓ %s R%d %s — %d drivers loaded", season, round_number, session_type, len(df)
    )
    return df


def fetch_pit_stops(season: int, round_number: int) -> pd.DataFrame | None:
    """
    Extract pit stop data for a race session.
    Returns per-constructor fastest stop and stop counts.

    FastF1 stores pit stop data across two consecutive laps per driver:
      - Lap N:   PitInTime  = when the car entered the pit lane
      - Lap N+1: PitOutTime = when the car exited the pit lane
    Duration = PitOutTime(N+1) - PitInTime(N), matched per driver.
    """
    try:
        session = fastf1.get_session(season, round_number, "Race")
        session.load(laps=True, telemetry=False, weather=False, messages=False)
    except Exception as exc:
        log.warning("Could not load pit stops R%d: %s", round_number, exc)
        return None

    laps = session.laps
    if laps is None or laps.empty:
        return None

    pit_records = []

    for driver in laps["Driver"].unique():
        driver_laps = (
            laps[laps["Driver"] == driver]
            .sort_values("LapNumber")
            .reset_index(drop=True)
        )

        for i, lap in driver_laps.iterrows():
            if pd.isna(lap["PitInTime"]):
                continue
            # PitOutTime lives on the very next lap for this driver
            next_laps = driver_laps[driver_laps["LapNumber"] == lap["LapNumber"] + 1]
            if next_laps.empty or pd.isna(next_laps.iloc[0]["PitOutTime"]):
                continue

            duration = (
                next_laps.iloc[0]["PitOutTime"] - lap["PitInTime"]
            ).total_seconds()

            if 1 <= duration <= 60:  # sanity filter — stationary time only
                pit_records.append(
                    {
                        "Driver": lap["Driver"],
                        "LapNumber": lap["LapNumber"],
                        "PitStopDuration": duration,
                    }
                )

    if not pit_records:
        log.warning(
            "  ✗ R%d PitStops — no valid stops found after duration filter",
            round_number,
        )
        return None

    pit_laps = pd.DataFrame(pit_records)

    constructor_map = (
        session.results[["Abbreviation", "TeamName"]]
        .rename(columns={"Abbreviation": "Driver", "TeamName": "Constructor"})
        .set_index("Driver")["Constructor"]
        .to_dict()
    )
    pit_laps["Constructor"] = pit_laps["Driver"].map(constructor_map)

    summary = (
        pit_laps.groupby("Constructor")
        .agg(
            FastestPitStop=("PitStopDuration", "min"),
            MeanPitStop=("PitStopDuration", "mean"),
            TotalPitStops=("PitStopDuration", "count"),
        )
        .reset_index()
    )

    summary["Season"] = season
    summary["Round"] = round_number

    # Flag the overall fastest pit stop (bonus points in F1 Fantasy)
    summary["FastestPitStopOfRace"] = (
        summary["FastestPitStop"] == summary["FastestPitStop"].min()
    )

    log.info(
        "  ✓ %s R%d PitStops — %d constructors", season, round_number, len(summary)
    )
    return summary


def fetch_weather(season: int, round_number: int) -> pd.DataFrame | None:
    """Fetch average weather conditions for the race session."""
    try:
        session = fastf1.get_session(season, round_number, "Race")
        session.load(laps=False, telemetry=False, weather=True, messages=False)
    except Exception as exc:
        log.warning("Could not load weather R%d: %s", round_number, exc)
        return None

    weather = session.weather_data
    if weather is None or weather.empty:
        return None

    summary = pd.DataFrame(
        [
            {
                "Season": season,
                "Round": round_number,
                "EventName": session.event["EventName"],
                "AvgAirTemp": weather["AirTemp"].mean(),
                "AvgTrackTemp": weather["TrackTemp"].mean(),
                "AvgHumidity": weather["Humidity"].mean(),
                "AvgWindSpeed": weather["WindSpeed"].mean(),
                "Rainfall": weather["Rainfall"].any(),
            }
        ]
    )

    log.info("  ✓ %s R%d Weather fetched", season, round_number)
    return summary


# ── DuckDB persistence ────────────────────────────────────────────────────────


def upsert_table(
    con: duckdb.DuckDBPyConnection, df: pd.DataFrame, table: str, pk_cols: list[str]
):
    """
    Create table if not exists, then insert rows that don't already exist
    (idempotent — safe to re-run).
    """
    con.execute(f"CREATE TABLE IF NOT EXISTS {table} AS SELECT * FROM df WHERE 1=0")

    existing_keys = con.execute(f"SELECT {', '.join(pk_cols)} FROM {table}").fetchdf()

    if existing_keys.empty:
        con.execute(f"INSERT INTO {table} SELECT * FROM df")
    else:
        merged = df.merge(existing_keys, on=pk_cols, how="left", indicator=True)
        new_rows = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])
        if not new_rows.empty:
            con.execute(f"INSERT INTO {table} SELECT * FROM new_rows")

    count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    log.info("    → %s: %d total rows in DB", table, count)


# ── Main ──────────────────────────────────────────────────────────────────────


def run(season: int, last_n: int | None = 3):
    log.info("━━━ F1 Fantasy Data Fetcher ━━━")
    log.info(
        "Season: %d | Fetching last %s rounds", season, last_n if last_n else "ALL"
    )

    rounds = get_completed_rounds(season)
    if not rounds:
        log.warning("No completed rounds found for %d", season)
        return

    if last_n:
        rounds = rounds[-last_n:]

    log.info("Rounds to fetch: %s", [r["RoundNumber"] for r in rounds])

    all_results: list[pd.DataFrame] = []
    all_pit_stops: list[pd.DataFrame] = []
    all_weather: list[pd.DataFrame] = []

    for r in rounds:
        rn = r["RoundNumber"]
        log.info("── Round %d: %s ──", rn, r["EventName"])

        for session_type in ["Race", "Qualifying", "Sprint"]:
            df = fetch_session(season, rn, session_type)
            if df is not None:
                all_results.append(df)

        pits = fetch_pit_stops(season, rn)
        if pits is not None:
            all_pit_stops.append(pits)

        wx = fetch_weather(season, rn)
        if wx is not None:
            all_weather.append(wx)

    # ── Persist to DuckDB ────────────────────────────────────────────────────
    con = duckdb.connect(str(DB_PATH))

    if all_results:
        results_df = pd.concat(
            [df.dropna(axis=1, how="all") for df in all_results], ignore_index=True
        )
        upsert_table(
            con,
            results_df,
            "session_results",
            ["Season", "Round", "SessionType", "Driver"],
        )
        log.info("Session results written: %d rows", len(results_df))

    if all_pit_stops:
        pits_df = pd.concat(
            [df.dropna(axis=1, how="all") for df in all_pit_stops], ignore_index=True
        )
        upsert_table(con, pits_df, "pit_stops", ["Season", "Round", "Constructor"])
        log.info("Pit stops written: %d rows", len(pits_df))

    if all_weather:
        weather_df = pd.concat(
            [df.dropna(axis=1, how="all") for df in all_weather], ignore_index=True
        )
        upsert_table(con, weather_df, "weather", ["Season", "Round"])
        log.info("Weather written: %d rows", len(weather_df))

    con.close()
    log.info("━━━ Done. DB saved to %s ━━━", DB_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch F1 race data into DuckDB")
    parser.add_argument("--season", type=int, default=2025, help="F1 season year")
    parser.add_argument(
        "--races", type=int, default=3, help="Number of most recent races to fetch"
    )
    parser.add_argument("--full", action="store_true", help="Fetch entire season")
    args = parser.parse_args()

    run(
        season=args.season,
        last_n=None if args.full else args.races,
    )
