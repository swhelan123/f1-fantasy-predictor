"""
engineer.py — 2026 SCORING (corrected)

Reads raw session results, pit stop, and weather data from DuckDB and produces
a fully-featured ML-ready dataset, also stored in DuckDB.

2026 scoring fixes vs previous version:
  - Positions gained: 1pt per position (not 2)
  - Race fastest lap: 10pts (not 5)
  - Race DNF penalty: -20 (not -15)
  - Pit bands: <2.0=20, 2.0-2.19=10, 2.20-2.49=5, 2.50-2.99=2, >=3.0=0
  - Qualifying NC/DSQ: -5pts
  - Constructor Q2/Q3 progression points
  - Sprint DNF: -10 (halved in 2026)

Usage:
    python src/features/engineer.py
"""

import logging
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "f1_fantasy.duckdb"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── 2026 Scoring Constants ────────────────────────────────────────────────────

RACE_FINISH_PTS = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
QUALI_PTS = {1: 10, 2: 9, 3: 8, 4: 7, 5: 6, 6: 5, 7: 4, 8: 3, 9: 2, 10: 1}
SPRINT_FINISH_PTS = {1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1}

POS_GAINED_PTS = 1  # per position gained (race + sprint)
POS_LOST_PTS = -1  # per position lost
RACE_FASTEST_LAP = 10
RACE_DNF_PENALTY = -20
SPRINT_DNF_PENALTY = -10  # halved in 2026
QUALI_NC_PENALTY = -5  # NC / no time set in Q1

TEAMMATE_BEAT_BONUS = 3  # beat teammate in quali or race

# Constructor Q2/Q3 progression
CTOR_PROG = {
    "neither_q2": -1,
    "one_q2": 1,
    "both_q2": 3,
    "one_q3": 5,
    "both_q3": 10,
}


# Pit stop bands (2026 official)
def pit_stop_pts(t: float) -> int:
    if pd.isna(t):
        return 0
    if t < 2.00:
        return 20
    if t < 2.20:
        return 10
    if t < 2.50:
        return 5
    if t < 3.00:
        return 2
    return 0


FASTEST_PIT_BONUS = 5
PIT_WORLD_RECORD_BONUS = 15  # replaces +5 if new world record
PIT_WORLD_RECORD_TIME = 1.80  # McLaren Qatar 2023


# ── Driver Points Per Session ─────────────────────────────────────────────────


def driver_race_pts(df: pd.DataFrame) -> pd.Series:
    pts = pd.Series(0.0, index=df.index)
    pos = pd.to_numeric(df["Position"], errors="coerce")
    grid = pd.to_numeric(df["GridPosition"], errors="coerce")

    pts += pos.map(RACE_FINISH_PTS).fillna(0)

    gained = (grid - pos).fillna(0)
    dnf = (
        df["DNF"].astype(bool)
        if "DNF" in df.columns
        else pd.Series(False, index=df.index)
    )

    # Positions gained/lost only for classified finishers
    pts[~dnf] += gained[~dnf].clip(lower=0) * POS_GAINED_PTS
    pts[~dnf] += gained[~dnf].clip(upper=0) * abs(POS_LOST_PTS)

    # DNF flat penalty (no positions-lost added)
    pts[dnf] += RACE_DNF_PENALTY

    if "FastestLap" in df.columns:
        pts += df["FastestLap"].astype(bool).astype(int) * RACE_FASTEST_LAP

    return pts


def driver_quali_pts(df: pd.DataFrame) -> pd.Series:
    pts = pd.Series(0.0, index=df.index)
    pos = pd.to_numeric(df["Position"], errors="coerce")
    pts += pos.map(QUALI_PTS).fillna(0)
    if "Status" in df.columns:
        nc = df["Status"].str.contains("no time|NC|DSQ", case=False, na=False)
        pts[nc] += QUALI_NC_PENALTY
    return pts


def driver_sprint_pts(df: pd.DataFrame) -> pd.Series:
    pts = pd.Series(0.0, index=df.index)
    pos = pd.to_numeric(df["Position"], errors="coerce")
    grid = pd.to_numeric(df["GridPosition"], errors="coerce")

    pts += pos.map(SPRINT_FINISH_PTS).fillna(0)

    gained = (grid - pos).fillna(0)
    dnf = (
        df["DNF"].astype(bool)
        if "DNF" in df.columns
        else pd.Series(False, index=df.index)
    )

    pts[~dnf] += gained[~dnf].clip(lower=0) * POS_GAINED_PTS
    pts[~dnf] += gained[~dnf].clip(upper=0) * abs(POS_LOST_PTS)
    pts[dnf] += SPRINT_DNF_PENALTY

    return pts


def teammate_bonus(df: pd.DataFrame) -> pd.Series:
    """Award +3 to the driver who finishes ahead of their teammate."""
    bonus = pd.Series(0.0, index=df.index)
    pos = pd.to_numeric(df["Position"], errors="coerce")
    for ctor in df["Constructor"].unique():
        mask = df["Constructor"] == ctor
        team_pos = pos[mask].dropna()
        if len(team_pos) < 2:
            continue
        bonus.loc[team_pos.idxmin()] = TEAMMATE_BEAT_BONUS
    return bonus


# ── Constructor Quali Progression ─────────────────────────────────────────────


def ctor_quali_progression(quali_df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame with Constructor + QualiProgressionPts."""
    rows = []
    pos = pd.to_numeric(quali_df["Position"], errors="coerce")
    for ctor in quali_df["Constructor"].unique():
        team_pos = pos[quali_df["Constructor"] == ctor].dropna().tolist()
        q3 = sum(1 for p in team_pos if p <= 10)
        q2 = sum(1 for p in team_pos if p <= 15)
        if q3 == 2:
            key = "both_q3"
        elif q3 == 1:
            key = "one_q3"
        elif q2 == 2:
            key = "both_q2"
        elif q2 == 1:
            key = "one_q2"
        else:
            key = "neither_q2"
        rows.append({"Constructor": ctor, "QualiProgressionPts": CTOR_PROG[key]})
    return pd.DataFrame(rows)


# ── Load Raw ──────────────────────────────────────────────────────────────────


def load_raw(con):
    results = con.execute(
        "SELECT * FROM session_results ORDER BY Season, Round"
    ).fetchdf()
    pit_stops = con.execute("SELECT * FROM pit_stops ORDER BY Season, Round").fetchdf()
    weather = con.execute("SELECT * FROM weather ORDER BY Season, Round").fetchdf()
    log.info(
        "Raw: %d results | %d pit stops | %d weather",
        len(results),
        len(pit_stops),
        len(weather),
    )
    return results, pit_stops, weather


# ── Step 1: Weekend Fantasy Points ───────────────────────────────────────────


def build_session_points(results: pd.DataFrame) -> pd.DataFrame:
    log.info("Step 1 — computing fantasy points...")

    race_df = results[results["SessionType"] == "Race"].copy()
    quali_df = results[results["SessionType"] == "Qualifying"].copy()
    sprint_df = results[results["SessionType"] == "Sprint"].copy()

    race_df["SessionPts"] = driver_race_pts(race_df) + teammate_bonus(race_df)
    quali_df["SessionPts"] = driver_quali_pts(quali_df) + teammate_bonus(quali_df)
    if not sprint_df.empty:
        sprint_df["SessionPts"] = driver_sprint_pts(sprint_df) + teammate_bonus(
            sprint_df
        )

    # Base from race
    base = race_df[
        [
            "Season",
            "Round",
            "Driver",
            "Constructor",
            "EventName",
            "Location",
            "Position",
            "GridPosition",
            "DNF",
            "FastestLap",
            "SessionPts",
        ]
    ].rename(columns={"Position": "RacePosition", "SessionPts": "RaceFantasyPts"})

    # Join quali
    qj = quali_df[["Season", "Round", "Driver", "Position", "SessionPts"]].rename(
        columns={"Position": "QualifyingPosition", "SessionPts": "QualiFantasyPts"}
    )
    base = base.merge(qj, on=["Season", "Round", "Driver"], how="left")

    # Join sprint
    if not sprint_df.empty:
        sj = sprint_df[
            ["Season", "Round", "Driver", "Position", "DNF", "SessionPts"]
        ].rename(
            columns={
                "Position": "SprintPosition",
                "DNF": "SprintDNF",
                "SessionPts": "SprintFantasyPts",
            }
        )
        base = base.merge(sj, on=["Season", "Round", "Driver"], how="left")
    else:
        base["SprintPosition"] = np.nan
        base["SprintDNF"] = False
        base["SprintFantasyPts"] = 0.0

    base["QualiFantasyPts"] = base["QualiFantasyPts"].fillna(0.0)
    base["SprintFantasyPts"] = base["SprintFantasyPts"].fillna(0.0)
    base["TotalFantasyPts"] = (
        base["RaceFantasyPts"] + base["QualiFantasyPts"] + base["SprintFantasyPts"]
    )
    base["HasSprint"] = base["SprintPosition"].notna()

    for c in ["RacePosition", "GridPosition", "QualifyingPosition", "SprintPosition"]:
        base[c] = pd.to_numeric(base[c], errors="coerce")

    log.info(
        "  %d driver-weekend rows | pts %.1f – %.1f (mean %.1f)",
        len(base),
        base["TotalFantasyPts"].min(),
        base["TotalFantasyPts"].max(),
        base["TotalFantasyPts"].mean(),
    )
    return base


# ── Step 2: Rolling Form ──────────────────────────────────────────────────────


def add_rolling(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    log.info("Step 2 — rolling form (window=%d)...", window)
    df = df.sort_values(["Driver", "Season", "Round"]).reset_index(drop=True)
    for drv, grp in df.groupby("Driver"):
        idx = grp.index
        r = lambda s: s.shift(1).rolling(window, min_periods=1).mean()
        df.loc[idx, "RollingAvgFinish"] = r(grp["RacePosition"])
        df.loc[idx, "RollingAvgFantasyPts"] = r(grp["TotalFantasyPts"])
        df.loc[idx, "RollingDNFRate"] = r(grp["DNF"].astype(float))
        df.loc[idx, "RollingAvgPositionsGained"] = r(
            (grp["GridPosition"] - grp["RacePosition"]).fillna(0)
        )
        df.loc[idx, "RollingAvgQualiPos"] = r(grp["QualifyingPosition"])
    return df


# ── Step 3: Circuit History ───────────────────────────────────────────────────


def add_circuit(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Step 3 — circuit history (may take a moment)...")
    df = df.sort_values(["Driver", "Location", "Season", "Round"]).reset_index(
        drop=True
    )
    avg_finish, dnf_rate, avg_pts = [], [], []
    for _, row in df.iterrows():
        prior = df[
            (df["Driver"] == row["Driver"])
            & (df["Location"] == row["Location"])
            & (
                (df["Season"] < row["Season"])
                | ((df["Season"] == row["Season"]) & (df["Round"] < row["Round"]))
            )
        ]
        avg_finish.append(prior["RacePosition"].mean() if len(prior) else np.nan)
        dnf_rate.append(prior["DNF"].astype(float).mean() if len(prior) else np.nan)
        avg_pts.append(prior["TotalFantasyPts"].mean() if len(prior) else np.nan)
    df["CircuitAvgFinish"] = avg_finish
    df["CircuitDNFRate"] = dnf_rate
    df["CircuitAvgFantasyPts"] = avg_pts
    return df


# ── Step 4: Teammate Deltas ───────────────────────────────────────────────────


def add_teammate(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Step 4 — teammate deltas...")
    df = df.copy()
    df["QualiVsTeammate"] = np.nan
    df["RaceVsTeammate"] = np.nan
    df["AvgTeammateFantasyPts"] = np.nan
    for (s, r, c), grp in df.groupby(["Season", "Round", "Constructor"]):
        if len(grp) < 2:
            continue
        for idx in grp.index:
            tm = grp[grp.index != idx].iloc[0]
            qs, qt = grp.loc[idx, "QualifyingPosition"], tm["QualifyingPosition"]
            if pd.notna(qs) and pd.notna(qt):
                df.loc[idx, "QualiVsTeammate"] = qt - qs
            rs, rt = grp.loc[idx, "RacePosition"], tm["RacePosition"]
            if pd.notna(rs) and pd.notna(rt):
                df.loc[idx, "RaceVsTeammate"] = rt - rs
            df.loc[idx, "AvgTeammateFantasyPts"] = grp[grp.index != idx][
                "TotalFantasyPts"
            ].mean()
    return df


# ── Step 5: Pit Stop Features ─────────────────────────────────────────────────


def add_pitstops(df: pd.DataFrame, pit_stops: pd.DataFrame) -> pd.DataFrame:
    log.info("Step 5 — pit stop features...")
    ps = pit_stops.copy()
    ps["PitStopFantasyPts"] = ps["FastestPitStop"].apply(pit_stop_pts)
    ps.loc[ps["FastestPitStopOfRace"], "PitStopFantasyPts"] += FASTEST_PIT_BONUS
    ps.loc[ps["FastestPitStop"] < PIT_WORLD_RECORD_TIME, "PitStopFantasyPts"] += (
        PIT_WORLD_RECORD_BONUS - FASTEST_PIT_BONUS
    )

    ps = ps.sort_values(["Constructor", "Season", "Round"])
    ps["RollingAvgPitStop"] = ps.groupby("Constructor")["FastestPitStop"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
    ps["RollingPitStopStd"] = ps.groupby("Constructor")["FastestPitStop"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).std()
    )

    df = df.merge(
        ps[
            [
                "Season",
                "Round",
                "Constructor",
                "FastestPitStop",
                "PitStopFantasyPts",
                "RollingAvgPitStop",
                "RollingPitStopStd",
                "FastestPitStopOfRace",
            ]
        ],
        on=["Season", "Round", "Constructor"],
        how="left",
    )
    return df


# ── Step 6: Weather ───────────────────────────────────────────────────────────


def add_weather(df: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    log.info("Step 6 — weather features...")
    df = df.merge(
        weather[
            [
                "Season",
                "Round",
                "AvgAirTemp",
                "AvgTrackTemp",
                "AvgHumidity",
                "AvgWindSpeed",
                "Rainfall",
            ]
        ],
        on=["Season", "Round"],
        how="left",
    )
    df["Rainfall"] = df["Rainfall"].fillna(False).astype(int)
    return df


# ── Step 7: PPM Proxy ─────────────────────────────────────────────────────────


def add_ppm(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Step 7 — PPM proxy...")
    df["RollingPPM_Proxy"] = df.groupby("Driver")["TotalFantasyPts"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
    return df


# ── Step 8: Encode Categoricals ──────────────────────────────────────────────


def encode(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Step 8 — encoding categoricals...")
    for col in ["Driver", "Constructor", "Location", "EventName"]:
        df[col + "_enc"] = df[col].astype("category").cat.codes
    return df


# ── Feature Columns ───────────────────────────────────────────────────────────

FEATURE_COLS = [
    "Season",
    "Round",
    "Driver",
    "Constructor",
    "EventName",
    "Location",
    "TotalFantasyPts",
    "RacePosition",
    "GridPosition",
    "QualifyingPosition",
    "DNF",
    "FastestLap",
    "HasSprint",
    "RollingAvgFinish",
    "RollingAvgFantasyPts",
    "RollingDNFRate",
    "RollingAvgPositionsGained",
    "RollingAvgQualiPos",
    "CircuitAvgFinish",
    "CircuitDNFRate",
    "CircuitAvgFantasyPts",
    "QualiVsTeammate",
    "RaceVsTeammate",
    "AvgTeammateFantasyPts",
    "FastestPitStop",
    "PitStopFantasyPts",
    "RollingAvgPitStop",
    "RollingPitStopStd",
    "FastestPitStopOfRace",
    "AvgAirTemp",
    "AvgTrackTemp",
    "AvgHumidity",
    "AvgWindSpeed",
    "Rainfall",
    "RollingPPM_Proxy",
    "Driver_enc",
    "Constructor_enc",
    "Location_enc",
]


# ── Main ──────────────────────────────────────────────────────────────────────


def run():
    log.info("━━━ F1 Fantasy Feature Engineering (2026 scoring) ━━━")
    con = duckdb.connect(str(DB_PATH))
    results, pit_stops, weather = load_raw(con)

    df = build_session_points(results)
    df = add_rolling(df)
    df = add_circuit(df)
    df = add_teammate(df)
    df = add_pitstops(df, pit_stops)
    df = add_weather(df, weather)
    df = add_ppm(df)
    df = encode(df)

    final = df[[c for c in FEATURE_COLS if c in df.columns]].copy()

    log.info("Final: %d rows × %d cols", len(final), len(final.columns))

    con.execute("DROP TABLE IF EXISTS features")
    con.execute("CREATE TABLE features AS SELECT * FROM final")
    final.to_parquet(PROCESSED_DIR / "features.parquet", index=False)

    con.close()
    log.info("━━━ Done ━━━")
    return final


if __name__ == "__main__":
    run()
