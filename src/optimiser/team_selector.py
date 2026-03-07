"""
team_selector.py

Selects the optimal F1 Fantasy team using PuLP integer linear programming.

2026 F1 Fantasy constraints:
  - Budget: $100M
  - Squad: 5 drivers + 2 constructors
  - Max 2 drivers from same constructor
  - 1 driver designated as "Turbo Driver" (points doubled)
  - Price floor: $3M

Transfer logic (for race-to-race changes):
  - 3 free transfers per race week
  - 1 unused transfer carries over (max 4)
  - Net transfers only (revert = no penalty)

Usage:
    python src/optimiser/team_selector.py
    python src/optimiser/team_selector.py --budget 100 --turbo VER
    python src/optimiser/team_selector.py --current-team VER,NOR,LEC,RUS,PIA,McLaren,Ferrari
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
from pulp import (
    PULP_CBC_CMD,
    LpBinary,
    LpMaximize,
    LpProblem,
    LpVariable,
    lpSum,
    value,
)

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── 2026 Driver List & Prices ─────────────────────────────────────────────────
# Prices in $M — update weekly from scrape_prices.py once built.
# These are approximate pre-season launch prices.

DRIVER_PRICES_2026 = {
    "VER": 27.7,
    "NOR": 27.2,
    "LEC": 22.8,
    "PIA": 25.5,
    "RUS": 27.4,
    "HAM": 22.5,
    "ANT": 23.2,
    "ALO": 10.0,
    "STR": 9.0,
    "SAI": 11.8,
    "ALB": 11.6,
    "HUL": 6.8,
    "OCO": 7.3,
    "GAS": 12.0,
    "LAW": 6.5,
    "HAD": 15.1,
    "COL": 6.2,
    "BEA": 7.4,
    "LIN": 6.2,
    "BOT": 5.9,
    "PER": 6.0,
    "BOR": 6.4,
}

CONSTRUCTOR_PRICES_2026 = {
    "McLaren": 28.9,
    "Ferrari": 23.3,
    "Mercedes": 29.3,
    "Red Bull Racing": 24.0,
    "Aston Martin": 10.3,
    "Williams": 12.0,
    "Audi": 6.6,
    "Haas F1 Team": 7.4,
    "Alpine": 12.5,
    "Racing Bulls": 6.3,
    "Cadillac": 6.0,
}

# Current 2026 driver → constructor mapping
DRIVER_CONSTRUCTOR_2026 = {
    "VER": "Red Bull Racing",
    "HAD": "Red Bull Racing",
    "NOR": "McLaren",
    "PIA": "McLaren",
    "LEC": "Ferrari",
    "HAM": "Ferrari",
    "RUS": "Mercedes",
    "ANT": "Mercedes",
    "ALO": "Aston Martin",
    "STR": "Aston Martin",
    "SAI": "Williams",
    "ALB": "Williams",
    "HUL": "Audi",
    "BOR": "Audi",
    "OCO": "Haas F1 Team",
    "BEA": "Haas F1 Team",
    "GAS": "Alpine",
    "COL": "Alpine",
    "LAW": "Racing Bulls",
    "LIN": "Racing Bulls",
    "BOT": "Cadillac",
    "PER": "Cadillac",
}


# ── Load Predictions ──────────────────────────────────────────────────────────


def load_predictions(
    season: int | None = None,
    round_number: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load driver and constructor prediction parquets.

    If *season* and *round_number* are given, load the round-specific files.
    Otherwise fall back to the ``_latest`` files.
    """
    if season is not None and round_number is not None:
        driver_path = (
            PROCESSED / f"predictions_drivers_S{season}_R{round_number:02d}.parquet"
        )
        ctor_path = (
            PROCESSED
            / f"predictions_constructors_S{season}_R{round_number:02d}.parquet"
        )
    else:
        driver_path = PROCESSED / "predictions_drivers_latest.parquet"
        ctor_path = PROCESSED / "predictions_constructors_latest.parquet"

    if not driver_path.exists():
        raise FileNotFoundError(
            f"No predictions found at {driver_path}. Run predict.py first."
        )

    drivers = pd.read_parquet(driver_path)
    ctors = pd.read_parquet(ctor_path)
    return drivers, ctors


# ── Enrich with Prices ────────────────────────────────────────────────────────


def enrich_with_prices(
    drivers: pd.DataFrame,
    ctors: pd.DataFrame,
    driver_prices: dict | None = None,
    ctor_prices: dict | None = None,
    exclude_drivers: list[str] | None = None,
    exclude_ctors: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Attach prices to prediction dataframes.
    Falls back to hardcoded 2026 launch prices if scraper hasn't run.
    """
    dp = driver_prices or DRIVER_PRICES_2026
    cp = ctor_prices or CONSTRUCTOR_PRICES_2026

    # Normalise constructor names (Kick Sauber → Audi for 2026)
    drivers["Constructor"] = drivers["Constructor"].replace({"Kick Sauber": "Audi"})

    # Filter to known 2026 drivers only (removes ghost drivers from old seasons)
    drivers = drivers[drivers["Driver"].isin(dp)].copy()
    ctors = ctors[ctors["Constructor"].isin(cp)].copy()

    # Apply exclusions
    if exclude_drivers:
        drivers = drivers[~drivers["Driver"].isin(exclude_drivers)]
        log.info("Excluded drivers: %s", exclude_drivers)
    if exclude_ctors:
        ctors = ctors[~ctors["Constructor"].isin(exclude_ctors)]
        log.info("Excluded constructors: %s", exclude_ctors)

    drivers["Price"] = drivers["Driver"].map(dp)
    ctors["Price"] = ctors["Constructor"].map(cp)

    # PPM = predicted points per million
    drivers["PPM"] = drivers["PredictedPts"] / drivers["Price"]
    ctors["PPM"] = ctors["PredictedPts"] / ctors["Price"]

    log.info(
        "Enriched %d drivers, %d constructors with prices", len(drivers), len(ctors)
    )
    return drivers, ctors


# ── PuLP Optimiser ────────────────────────────────────────────────────────────


def optimise_team(
    drivers: pd.DataFrame,
    ctors: pd.DataFrame,
    budget: float = 100.0,
    turbo_driver: str | None = None,
    current_team: list[str] | None = None,
    free_transfers: int = 3,
) -> dict:
    """
    Solve the F1 Fantasy team selection as an integer linear programme.

    Decision variables:
      d[i] ∈ {0,1}  — is driver i selected?
      c[j] ∈ {0,1}  — is constructor j selected?
      t[i] ∈ {0,1}  — is driver i the turbo driver? (points × 2)

    Objective: maximise sum(d[i] * pts[i]) + sum(t[i] * pts[i]) + sum(c[j] * pts[j])
      (turbo driver's points counted twice = base + bonus)

    Constraints:
      - Exactly 5 drivers selected
      - Exactly 2 constructors selected
      - Exactly 1 turbo driver
      - Total price ≤ budget
      - Max 2 drivers per constructor
      - Turbo driver must be in selected team
      - Transfer penalty if current_team provided
    """
    log.info("Running PuLP optimiser (budget $%.1fM)...", budget)

    prob = LpProblem("F1_Fantasy_Team", LpMaximize)

    # ── Decision variables ────────────────────────────────────────────────────
    driver_list = drivers["Driver"].tolist()
    ctor_list = ctors["Constructor"].tolist()

    d = {drv: LpVariable(f"d_{drv}", cat=LpBinary) for drv in driver_list}
    c = {
        ctor: LpVariable(f"c_{ctor.replace(' ', '_')}", cat=LpBinary)
        for ctor in ctor_list
    }

    # Turbo driver variable — if pre-specified, fix it; otherwise optimise
    if turbo_driver and turbo_driver in driver_list:
        # Force turbo to the specified driver
        t = {drv: LpVariable(f"t_{drv}", cat=LpBinary) for drv in driver_list}
        prob += t[turbo_driver] == 1
        for drv in driver_list:
            if drv != turbo_driver:
                prob += t[drv] == 0
    else:
        t = {drv: LpVariable(f"t_{drv}", cat=LpBinary) for drv in driver_list}

    # ── Objective ─────────────────────────────────────────────────────────────
    pts = dict(zip(drivers["Driver"], drivers["PredictedPts"]))
    ctor_pts = dict(zip(ctors["Constructor"], ctors["PredictedPts"]))
    prices = dict(zip(drivers["Driver"], drivers["Price"]))
    ctor_prices = dict(zip(ctors["Constructor"], ctors["Price"]))

    # Transfer penalty — each transfer beyond free allowance costs ~4 pts
    transfer_penalty = 0
    if current_team:
        current_drivers = set(current_team[:5])
        current_ctors = set(current_team[5:])
        # Count how many changes are made
        n_driver_changes = lpSum(
            d[drv] for drv in driver_list if drv not in current_drivers
        )
        n_ctor_changes = lpSum(
            c[ctor] for ctor in ctor_list if ctor not in current_ctors
        )
        total_changes = n_driver_changes + n_ctor_changes
        excess_transfers = total_changes - free_transfers
        # Penalise excess transfers at 10 pts each (2026 rules)
        transfer_penalty = 10 * excess_transfers

    prob += (
        lpSum(pts[drv] * d[drv] for drv in driver_list)
        + lpSum(pts[drv] * t[drv] for drv in driver_list)  # turbo bonus
        + lpSum(ctor_pts[ctor] * c[ctor] for ctor in ctor_list)
        - transfer_penalty
    )

    # ── Constraints ───────────────────────────────────────────────────────────

    # Exactly 5 drivers, 2 constructors
    prob += lpSum(d[drv] for drv in driver_list) == 5
    prob += lpSum(c[ctor] for ctor in ctor_list) == 2

    # Exactly 1 turbo driver, and turbo must be in selected team
    prob += lpSum(t[drv] for drv in driver_list) == 1
    for drv in driver_list:
        prob += t[drv] <= d[drv]  # can't turbo a driver not in team

    # Budget constraint
    prob += (
        lpSum(prices[drv] * d[drv] for drv in driver_list)
        + lpSum(ctor_prices[ctor] * c[ctor] for ctor in ctor_list)
        <= budget
    )

    # Max 2 drivers per constructor
    ctor_map = DRIVER_CONSTRUCTOR_2026
    for ctor in ctor_list:
        ctor_drivers = [drv for drv in driver_list if ctor_map.get(drv) == ctor]
        if ctor_drivers:
            prob += lpSum(d[drv] for drv in ctor_drivers) <= 2

    # ── Solve ─────────────────────────────────────────────────────────────────
    prob.solve(PULP_CBC_CMD(msg=0))

    # ── Extract solution ──────────────────────────────────────────────────────
    selected_drivers = [drv for drv in driver_list if value(d[drv]) == 1]
    selected_ctors = [ctor for ctor in ctor_list if value(c[ctor]) == 1]
    turbo = next((drv for drv in driver_list if value(t[drv]) == 1), None)

    total_cost = sum(prices[drv] for drv in selected_drivers) + sum(
        ctor_prices[ctor] for ctor in selected_ctors
    )

    # Predicted score with turbo
    predicted_score = (
        sum(pts[drv] for drv in selected_drivers)
        + pts.get(turbo, 0)  # turbo bonus
        + sum(ctor_pts[ctor] for ctor in selected_ctors)
    )

    return {
        "drivers": selected_drivers,
        "constructors": selected_ctors,
        "turbo": turbo,
        "total_cost": total_cost,
        "budget_remaining": budget - total_cost,
        "predicted_score": predicted_score,
        "driver_details": drivers[drivers["Driver"].isin(selected_drivers)],
        "ctor_details": ctors[ctors["Constructor"].isin(selected_ctors)],
    }


# ── Display ───────────────────────────────────────────────────────────────────


def display_team(result: dict, drivers: pd.DataFrame, ctors: pd.DataFrame):
    """Print a clean team summary."""
    log.info("━━━ Optimal F1 Fantasy Team ━━━")

    # Drivers table
    d_detail = drivers[drivers["Driver"].isin(result["drivers"])].copy()
    d_detail = d_detail.sort_values("PredictedPts", ascending=False)
    d_detail["Turbo"] = d_detail["Driver"].apply(
        lambda x: "⚡" if x == result["turbo"] else ""
    )
    d_detail["Price"] = d_detail["Price"].apply(lambda x: f"${x:.1f}M")
    d_detail["PredictedPts"] = d_detail["PredictedPts"].round(1)
    d_detail["PPM"] = d_detail["PPM"].round(2)

    log.info("── Drivers ──")
    log.info(
        "\n%s",
        d_detail[
            ["Driver", "Constructor", "Price", "PredictedPts", "PPM", "Turbo"]
        ].to_string(index=False),
    )

    # Constructors table
    c_detail = ctors[ctors["Constructor"].isin(result["constructors"])].copy()
    c_detail["Price"] = c_detail["Price"].apply(lambda x: f"${x:.1f}M")
    c_detail["PredictedPts"] = c_detail["PredictedPts"].round(1)
    c_detail["PPM"] = c_detail["PPM"].round(2)

    log.info("── Constructors ──")
    log.info(
        "\n%s",
        c_detail[["Constructor", "Price", "PredictedPts", "PPM"]].to_string(
            index=False
        ),
    )

    log.info("── Summary ──")
    log.info("  Turbo Driver:    %s (pts doubled)", result["turbo"])
    log.info("  Total Cost:      $%.1fM", result["total_cost"])
    log.info("  Budget Remaining: $%.1fM", result["budget_remaining"])
    log.info("  Predicted Score: %.1f pts", result["predicted_score"])
    log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


def load_scraped_prices() -> tuple[dict | None, dict | None]:
    """
    Load prices from scrape_prices.py output if available.
    Returns (driver_prices, ctor_prices) dicts, or (None, None) to fall back to hardcoded.
    """
    prices_path = PROCESSED / "player_prices.parquet"
    if not prices_path.exists():
        log.info("No scraped prices found — using hardcoded launch prices")
        return None, None

    df = pd.read_parquet(prices_path)
    scraped_at = df["ScrapedAt"].max() if "ScrapedAt" in df.columns else "unknown"
    log.info("Using scraped prices from %s (%d records)", scraped_at, len(df))

    drivers_df = df[df["Type"].str.lower().str.contains("driver", na=False)]
    ctors_df = df[df["Type"].str.lower().str.contains("constructor", na=False)]

    driver_prices = dict(zip(drivers_df["Code"], drivers_df["Price"]))
    ctor_prices = dict(zip(ctors_df["Code"], ctors_df["Price"]))

    # Fall back to hardcoded if scraper returned too few results
    if len(driver_prices) < 10:
        log.warning(
            "Scraped driver prices too sparse (%d) — using hardcoded",
            len(driver_prices),
        )
        driver_prices = None
    if len(ctor_prices) < 5:
        log.warning(
            "Scraped constructor prices too sparse (%d) — using hardcoded",
            len(ctor_prices),
        )
        ctor_prices = None

    return driver_prices, ctor_prices


# ── Main ──────────────────────────────────────────────────────────────────────


def run(
    budget: float = 100.0,
    turbo_driver: str | None = None,
    current_team: list[str] | None = None,
    free_transfers: int = 3,
    exclude_drivers: list[str] | None = None,
    exclude_ctors: list[str] | None = None,
    season: int | None = None,
    round_number: int | None = None,
) -> dict:
    log.info("━━━ F1 Fantasy Team Selector ━━━")

    drivers, ctors = load_predictions(season=season, round_number=round_number)

    driver_prices, ctor_prices = load_scraped_prices()
    drivers, ctors = enrich_with_prices(
        drivers,
        ctors,
        driver_prices,
        ctor_prices,
        exclude_drivers=exclude_drivers,
        exclude_ctors=exclude_ctors,
    )

    log.info("── Prediction Summary ──")
    log.info(
        "\n%s",
        drivers[["Driver", "Constructor", "PredictedPts", "Price", "PPM"]]
        .sort_values("PredictedPts", ascending=False)
        .head(10)
        .round(2)
        .to_string(index=False),
    )

    result = optimise_team(
        drivers,
        ctors,
        budget=budget,
        turbo_driver=turbo_driver,
        current_team=current_team,
        free_transfers=free_transfers,
    )

    display_team(result, drivers, ctors)

    # Attach the full enriched dataframes so callers (e.g. report generator)
    # don't need to re-load and re-enrich predictions a second time.
    result["all_drivers"] = drivers
    result["all_ctors"] = ctors

    log.info("━━━ Done ━━━")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="F1 Fantasy team optimiser")
    parser.add_argument(
        "--budget", type=float, default=100.0, help="Budget in $M (default: 100)"
    )
    parser.add_argument(
        "--turbo",
        type=str,
        default=None,
        help="Force a specific turbo driver e.g. --turbo VER",
    )
    parser.add_argument(
        "--current-team",
        type=str,
        default=None,
        help="Comma-separated current team e.g. VER,NOR,LEC,RUS,PIA,McLaren,Ferrari",
    )
    parser.add_argument(
        "--transfers",
        type=int,
        default=3,
        help="Number of free transfers available (default: 3)",
    )
    parser.add_argument(
        "--exclude-drivers",
        type=str,
        default=None,
        help="Comma-separated drivers to exclude e.g. ALO,STR",
    )
    parser.add_argument(
        "--exclude-ctors",
        type=str,
        default=None,
        help="Comma-separated constructors to exclude e.g. 'Aston Martin'",
    )
    args = parser.parse_args()

    current = args.current_team.split(",") if args.current_team else None
    ex_drivers = args.exclude_drivers.split(",") if args.exclude_drivers else None
    ex_ctors = args.exclude_ctors.split(",") if args.exclude_ctors else None

    run(
        budget=args.budget,
        turbo_driver=args.turbo,
        current_team=current,
        free_transfers=args.transfers,
        exclude_drivers=ex_drivers,
        exclude_ctors=ex_ctors,
    )
