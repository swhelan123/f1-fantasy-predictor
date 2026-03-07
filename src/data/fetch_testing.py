"""
fetch_testing.py

Fetches pre-season testing data from FastF1 and extracts:
  - Fastest lap time per driver
  - Long run pace (avg of 10+ consecutive laps, best race pace proxy)
  - Total laps completed (reliability)
  - Tyre compound breakdown

Stores results in DuckDB table 'testing_results'.

Usage:
    python src/data/fetch_testing.py --season 2026
"""

import argparse
import logging
from pathlib import Path

import duckdb
import fastf1
import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "fastf1_cache"
DB_PATH = ROOT / "data" / "f1_fantasy.duckdb"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

fastf1.Cache.enable_cache(str(CACHE_DIR))

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Fetch One Testing Session ─────────────────────────────────────────────────


def fetch_test_session(
    season: int, test_number: int, session_number: int
) -> pd.DataFrame | None:
    """
    Fetch a single pre-season test session.
    test_number: which test event (usually 1)
    session_number: day 1/2/3 of testing
    """
    try:
        session = fastf1.get_testing_session(season, test_number, session_number)
        session.load(laps=True, telemetry=False, weather=False, messages=False)
        log.info("  ✓ Testing session %d day %d loaded", test_number, session_number)
    except Exception as exc:
        log.warning(
            "  Could not load test %d day %d: %s", test_number, session_number, exc
        )
        return None

    laps = session.laps
    if laps is None or laps.empty:
        return None

    # Filter to accurate laps only
    laps = laps[laps["IsAccurate"] == True].copy()
    if laps.empty:
        return None

    # Convert lap time to seconds
    laps["LapTimeSec"] = laps["LapTime"].dt.total_seconds()
    laps = laps[laps["LapTimeSec"].between(60, 200)]  # sanity filter

    # Map driver → constructor
    if session.results is not None and not session.results.empty:
        ctor_map = (
            session.results[["Abbreviation", "TeamName"]]
            .set_index("Abbreviation")["TeamName"]
            .to_dict()
        )
    else:
        ctor_map = {}

    rows = []
    for driver in laps["Driver"].unique():
        d_laps = laps[laps["Driver"] == driver].sort_values("LapNumber")

        # Fastest lap
        fastest = d_laps["LapTimeSec"].min()

        # Long run pace — find the longest consecutive stint of 10+ laps
        long_run_pace = _long_run_pace(d_laps)

        # Total laps
        total_laps = len(d_laps)

        rows.append(
            {
                "Season": season,
                "TestNumber": test_number,
                "SessionNumber": session_number,
                "Driver": driver,
                "Constructor": ctor_map.get(driver, "Unknown"),
                "FastestLapSec": fastest,
                "LongRunPaceSec": long_run_pace,
                "TotalLaps": total_laps,
            }
        )

    return pd.DataFrame(rows)


def _long_run_pace(driver_laps: pd.DataFrame, min_laps: int = 8) -> float:
    """
    Find the best (lowest avg) long run of at least min_laps consecutive laps.
    Consecutive = lap numbers are sequential with no gaps.
    """
    laps = driver_laps.sort_values("LapNumber").reset_index(drop=True)
    times = laps["LapTimeSec"].values
    lap_nums = laps["LapNumber"].values

    best_avg = np.nan
    current_run = [times[0]]

    for i in range(1, len(laps)):
        if lap_nums[i] == lap_nums[i - 1] + 1:
            current_run.append(times[i])
        else:
            if len(current_run) >= min_laps:
                avg = np.mean(current_run)
                if np.isnan(best_avg) or avg < best_avg:
                    best_avg = avg
            current_run = [times[i]]

    if len(current_run) >= min_laps:
        avg = np.mean(current_run)
        if np.isnan(best_avg) or avg < best_avg:
            best_avg = avg

    return best_avg


# ── Aggregate Across All Test Days ────────────────────────────────────────────


def aggregate_testing(all_sessions: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Aggregate across all test days per driver:
    - Best fastest lap
    - Best long run pace
    - Total laps across all days
    """
    df = pd.concat(all_sessions, ignore_index=True)

    agg = (
        df.groupby(["Season", "Driver", "Constructor"])
        .agg(
            TestFastestLap=("FastestLapSec", "min"),
            TestLongRunPace=("LongRunPaceSec", "min"),
            TestTotalLaps=("TotalLaps", "sum"),
        )
        .reset_index()
    )

    # Rank within season (1 = fastest)
    agg["TestFastestLapRank"] = agg["TestFastestLap"].rank().astype(int)
    agg["TestLongRunRank"] = agg["TestLongRunPace"].rank().astype(int)

    log.info("Testing summary: %d drivers", len(agg))
    log.info(
        "\n%s",
        agg[
            [
                "Driver",
                "Constructor",
                "TestFastestLap",
                "TestLongRunPace",
                "TestTotalLaps",
            ]
        ]
        .sort_values("TestFastestLap")
        .to_string(index=False),
    )

    return agg


# ── Persist ───────────────────────────────────────────────────────────────────


def save_to_db(df: pd.DataFrame):
    con = duckdb.connect(str(DB_PATH))
    con.execute("DROP TABLE IF EXISTS testing_results")
    con.execute("CREATE TABLE testing_results AS SELECT * FROM df")
    count = con.execute("SELECT COUNT(*) FROM testing_results").fetchone()[0]
    log.info("testing_results: %d rows saved to DB", count)
    con.close()


# ── Main ──────────────────────────────────────────────────────────────────────


def run(season: int = 2026, test_number: int = 1, num_days: int = 3):
    log.info("━━━ F1 Testing Data Fetcher (Season %d) ━━━", season)

    all_sessions = []
    for day in range(1, num_days + 1):
        df = fetch_test_session(season, test_number, day)
        if df is not None and not df.empty:
            all_sessions.append(df)

    if not all_sessions:
        log.warning("No testing data found for %d — skipping", season)
        return None

    agg = aggregate_testing(all_sessions)
    save_to_db(agg)

    log.info("━━━ Done ━━━")
    return agg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--test", type=int, default=1, help="Test event number")
    parser.add_argument("--days", type=int, default=3, help="Number of test days")
    args = parser.parse_args()
    run(season=args.season, test_number=args.test, num_days=args.days)
