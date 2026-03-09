"""
team_selector.py

Selects the optimal F1 Fantasy team using PuLP integer linear programming.

2026 F1 Fantasy constraints:
  - Budget: $100M
  - Squad: 5 drivers + 2 constructors
  - Max 2 drivers from same constructor
  - 1 driver designated as "Turbo Driver" (points doubled)
  - Price floor: $3M

Transfer logic (2026 rules):
  - 2 free transfers per race week
  - Excess transfers cost -10 pts each
  - When --current-team is provided, the optimiser finds the best team
    reachable within the free transfer limit, AND also shows the
    unconstrained optimum so you can see what you're leaving on the table.

Usage:
    # Greenfield (no current team — pick best unconstrained team):
    python src/optimiser/team_selector.py

    # With current team — find best team within 2 free transfers:
    python src/optimiser/team_selector.py --current-team VER,NOR,LEC,RUS,PIA,McLaren,Ferrari

    # Override budget or turbo:
    python src/optimiser/team_selector.py --budget 100 --turbo VER

    # Wildcard week — unlimited free transfers:
    python src/optimiser/team_selector.py --current-team VER,NOR,LEC,RUS,PIA,McLaren,Ferrari --free-transfers 7
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
    "STR": 8.0,
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
    "Red Bull Racing": 28.2,
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

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_FREE_TRANSFERS = 2  # 2026 rules: 2 free transfers per race week
EXCESS_TRANSFER_PENALTY = 10  # points deducted per excess transfer


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


# ── Parse Current Team String ─────────────────────────────────────────────────


def parse_current_team(team_str: str) -> tuple[list[str], list[str]]:
    """
    Parse a comma-separated current team string into drivers and constructors.

    The string should contain exactly 5 driver codes and 2 constructor names,
    in any order.  Constructors are identified by checking against the known
    constructor list; everything else is treated as a driver code.

    Example:
        "VER,NOR,LEC,RUS,PIA,McLaren,Ferrari"
        → (["VER", "NOR", "LEC", "RUS", "PIA"], ["McLaren", "Ferrari"])
    """
    all_ctors = set(CONSTRUCTOR_PRICES_2026.keys())
    # Build a lowercase/no-space → proper-case lookup for flexible input
    ctor_lower = {k.lower().replace(" ", ""): k for k in all_ctors}

    parts = [p.strip() for p in team_str.split(",") if p.strip()]

    drivers: list[str] = []
    constructors: list[str] = []

    for part in parts:
        # Check if it's a constructor (exact match or case-insensitive)
        if part in all_ctors:
            constructors.append(part)
        elif part.lower().replace(" ", "") in ctor_lower:
            constructors.append(ctor_lower[part.lower().replace(" ", "")])
        else:
            # Treat as driver code (uppercase it)
            drivers.append(part.upper())

    if len(drivers) != 5:
        log.warning(
            "Expected 5 drivers in current team, got %d: %s", len(drivers), drivers
        )
    if len(constructors) != 2:
        log.warning(
            "Expected 2 constructors in current team, got %d: %s",
            len(constructors),
            constructors,
        )

    return drivers, constructors


# ── Compute Transfer Diff ─────────────────────────────────────────────────────


def _compute_transfers(
    current_drivers: list[str],
    current_ctors: list[str],
    new_drivers: list[str],
    new_ctors: list[str],
    driver_prices: dict[str, float],
    ctor_prices: dict[str, float],
) -> tuple[list[dict], list[dict], int]:
    """Return (transfers_out, transfers_in, n_transfers)."""
    cur_drv = set(current_drivers)
    cur_ctor = set(current_ctors)
    new_drv = set(new_drivers)
    new_ctor = set(new_ctors)

    transfers_out: list[dict] = []
    transfers_in: list[dict] = []

    for drv in sorted(cur_drv - new_drv):
        transfers_out.append(
            {"name": drv, "type": "driver", "price": driver_prices.get(drv, 0)}
        )
    for drv in sorted(new_drv - cur_drv):
        transfers_in.append(
            {"name": drv, "type": "driver", "price": driver_prices.get(drv, 0)}
        )
    for ctor in sorted(cur_ctor - new_ctor):
        transfers_out.append(
            {"name": ctor, "type": "constructor", "price": ctor_prices.get(ctor, 0)}
        )
    for ctor in sorted(new_ctor - cur_ctor):
        transfers_in.append(
            {"name": ctor, "type": "constructor", "price": ctor_prices.get(ctor, 0)}
        )

    n_transfers = len(transfers_in)  # in == out always (team size is fixed)
    return transfers_out, transfers_in, n_transfers


# ── Core PuLP Optimiser ───────────────────────────────────────────────────────


def optimise_team(
    drivers: pd.DataFrame,
    ctors: pd.DataFrame,
    budget: float = 100.0,
    turbo_driver: str | None = None,
    current_team_drivers: list[str] | None = None,
    current_team_ctors: list[str] | None = None,
    free_transfers: int = DEFAULT_FREE_TRANSFERS,
    penalty_per_transfer: float = EXCESS_TRANSFER_PENALTY,
) -> dict | None:
    """
    Solve the F1 Fantasy team selection as an integer linear programme.

    Decision variables:
      d[i] ∈ {0,1}  — is driver i selected?
      c[j] ∈ {0,1}  — is constructor j selected?
      t[i] ∈ {0,1}  — is driver i the turbo driver? (points × 2)

    Objective:
      maximise  Σ(d[i]·pts[i]) + Σ(t[i]·pts[i]) + Σ(c[j]·pts[j])
                − penalty_per_transfer × max(0, n_changes − free_transfers)

      The penalty term is only active when a current team is provided.
      Transfers up to ``free_transfers`` are free; each additional transfer
      costs ``penalty_per_transfer`` points (default 10).  The solver will
      take extra transfers when the points gain exceeds the penalty cost.

    Constraints:
      - Exactly 5 drivers selected
      - Exactly 2 constructors selected
      - Exactly 1 turbo driver
      - Total price ≤ budget
      - Max 2 drivers per constructor
      - Turbo driver must be in selected team

    Returns:
        Result dict, or None if the problem is infeasible.
    """
    label = f"budget=${budget:.1f}M"
    if current_team_drivers is not None:
        label += f", free={free_transfers}, penalty={penalty_per_transfer}"
    log.info("Running PuLP optimiser (%s)...", label)

    prob = LpProblem("F1_Fantasy_Team", LpMaximize)

    # ── Decision variables ────────────────────────────────────────────────────
    driver_list = drivers["Driver"].tolist()
    ctor_list = ctors["Constructor"].tolist()

    d = {drv: LpVariable(f"d_{drv}", cat=LpBinary) for drv in driver_list}
    c = {
        ctor: LpVariable(f"c_{ctor.replace(' ', '_')}", cat=LpBinary)
        for ctor in ctor_list
    }

    # Turbo driver variable — if pre-specified, fix it; otherwise let solver pick
    t = {drv: LpVariable(f"t_{drv}", cat=LpBinary) for drv in driver_list}
    if turbo_driver and turbo_driver in driver_list:
        prob += t[turbo_driver] == 1
        for drv in driver_list:
            if drv != turbo_driver:
                prob += t[drv] == 0

    # ── Objective ─────────────────────────────────────────────────────────────
    pts = dict(zip(drivers["Driver"], drivers["PredictedPts"]))
    ctor_pts = dict(zip(ctors["Constructor"], ctors["PredictedPts"]))
    drv_prices = dict(zip(drivers["Driver"], drivers["Price"]))
    ctor_prices = dict(zip(ctors["Constructor"], ctors["Price"]))

    # ── Transfer penalty (soft) ───────────────────────────────────────────────
    # excess_transfers is a continuous variable ≥ 0 representing the number of
    # transfers beyond the free allowance.  It's subtracted from the objective
    # at ``penalty_per_transfer`` per unit, so the solver will only take extra
    # transfers when the points gain outweighs the cost.
    transfer_penalty_expr = 0
    if current_team_drivers is not None:
        current_drv_set = set(current_team_drivers)
        current_ctor_set = set(current_team_ctors or [])

        n_new_drivers = lpSum(
            d[drv] for drv in driver_list if drv not in current_drv_set
        )
        n_new_ctors = lpSum(
            c[ctor] for ctor in ctor_list if ctor not in current_ctor_set
        )
        total_changes = n_new_drivers + n_new_ctors

        # excess = max(0, total_changes - free_transfers)
        # Linearised via:  excess ≥ total_changes - free_transfers, excess ≥ 0
        excess = LpVariable("excess_transfers", lowBound=0, cat="Integer")
        prob += excess >= total_changes - free_transfers

        transfer_penalty_expr = penalty_per_transfer * excess

    prob += (
        lpSum(pts[drv] * d[drv] for drv in driver_list)
        + lpSum(pts[drv] * t[drv] for drv in driver_list)  # turbo bonus
        + lpSum(ctor_pts[ctor] * c[ctor] for ctor in ctor_list)
        - transfer_penalty_expr
    )

    # ── Constraints ───────────────────────────────────────────────────────────

    # Exactly 5 drivers, 2 constructors
    prob += lpSum(d[drv] for drv in driver_list) == 5
    prob += lpSum(c[ctor] for ctor in ctor_list) == 2

    # Exactly 1 turbo driver, and turbo must be selected
    prob += lpSum(t[drv] for drv in driver_list) == 1
    for drv in driver_list:
        prob += t[drv] <= d[drv]

    # Budget constraint
    prob += (
        lpSum(drv_prices[drv] * d[drv] for drv in driver_list)
        + lpSum(ctor_prices[ctor] * c[ctor] for ctor in ctor_list)
        <= budget
    )

    # Max 2 drivers per constructor
    ctor_map = DRIVER_CONSTRUCTOR_2026
    for ctor in ctor_list:
        team_drivers = [drv for drv in driver_list if ctor_map.get(drv) == ctor]
        if team_drivers:
            prob += lpSum(d[drv] for drv in team_drivers) <= 2

    # ── Solve ─────────────────────────────────────────────────────────────────
    prob.solve(PULP_CBC_CMD(msg=0))

    if prob.status != 1:
        log.warning(
            "Optimiser status %d (infeasible/unbounded)",
            prob.status,
        )
        return None

    # ── Extract solution ──────────────────────────────────────────────────────
    selected_drivers = [drv for drv in driver_list if value(d[drv]) == 1]
    selected_ctors = [ctor for ctor in ctor_list if value(c[ctor]) == 1]
    turbo = next((drv for drv in driver_list if value(t[drv]) == 1), None)

    total_cost = sum(drv_prices[drv] for drv in selected_drivers) + sum(
        ctor_prices[ctor] for ctor in selected_ctors
    )

    # Gross predicted score (before penalty)
    gross_score = (
        sum(pts[drv] for drv in selected_drivers)
        + pts.get(turbo, 0)  # turbo bonus
        + sum(ctor_pts[ctor] for ctor in selected_ctors)
    )

    # Transfer diff
    transfers_out: list[dict] = []
    transfers_in: list[dict] = []
    n_transfers = 0
    transfer_cost = 0.0
    excess_count = 0
    if current_team_drivers is not None:
        transfers_out, transfers_in, n_transfers = _compute_transfers(
            current_team_drivers,
            current_team_ctors or [],
            selected_drivers,
            selected_ctors,
            drv_prices,
            ctor_prices,
        )
        excess_count = max(0, n_transfers - free_transfers)
        transfer_cost = excess_count * penalty_per_transfer

    net_score = gross_score - transfer_cost

    return {
        "drivers": selected_drivers,
        "constructors": selected_ctors,
        "turbo": turbo,
        "total_cost": total_cost,
        "budget_remaining": budget - total_cost,
        "predicted_score": net_score,
        "gross_score": gross_score,
        "transfer_cost": transfer_cost,
        "excess_transfers": excess_count,
        "driver_details": drivers[drivers["Driver"].isin(selected_drivers)],
        "ctor_details": ctors[ctors["Constructor"].isin(selected_ctors)],
        "n_transfers": n_transfers,
        "transfers_in": transfers_in,
        "transfers_out": transfers_out,
        "free_transfers": free_transfers,
        "penalty_per_transfer": penalty_per_transfer,
    }


# ── Transfer-Aware Optimisation ───────────────────────────────────────────────


def optimise_with_transfers(
    drivers: pd.DataFrame,
    ctors: pd.DataFrame,
    budget: float = 100.0,
    turbo_driver: str | None = None,
    current_team_drivers: list[str] | None = None,
    current_team_ctors: list[str] | None = None,
    free_transfers: int = DEFAULT_FREE_TRANSFERS,
) -> dict:
    """
    Find the best team given a current team and a free-transfer budget.

    The solver uses a **soft penalty**: transfers beyond ``free_transfers``
    each cost 10 pts in the objective, so extra transfers are only taken when
    the points gain outweighs the penalty.  This means the solver might
    recommend 0, 1, 2, or even 5 transfers if that nets the highest score.

    Additionally an unconstrained solve (no current team, no penalty) is run
    to show the theoretical ceiling so the user can plan towards it.

    If no current team is provided, runs a single unconstrained solve.

    Extra keys on the returned dict:
      - ``"current_team_drivers"`` / ``"current_team_ctors"`` — echo of input
      - ``"unconstrained_result"``— the result with no transfer penalty at all
      - ``"gross_delta"``         — gross-score gap to unconstrained optimum
    """
    # ── No current team → single unconstrained solve ──────────────────────────
    if current_team_drivers is None:
        log.info("No current team provided — running unconstrained optimisation")
        result = optimise_team(
            drivers,
            ctors,
            budget=budget,
            turbo_driver=turbo_driver,
        )
        if result is None:
            raise RuntimeError("Unconstrained optimisation was infeasible — check data")
        result["current_team_drivers"] = None
        result["current_team_ctors"] = None
        result["unconstrained_result"] = None
        result["gross_delta"] = 0.0
        return result

    # ── Current team provided ─────────────────────────────────────────────────
    log.info(
        "Current team: drivers=%s, constructors=%s",
        current_team_drivers,
        current_team_ctors,
    )
    log.info(
        "Free transfers: %d  |  Penalty per excess: %d pts",
        free_transfers,
        EXCESS_TRANSFER_PENALTY,
    )

    # 1) Penalised solve — the *actual* recommendation.
    #    The solver sees the 10-pt penalty in the objective so it naturally
    #    balances "more transfers = more raw pts but higher cost".
    penalised = optimise_team(
        drivers,
        ctors,
        budget=budget,
        turbo_driver=turbo_driver,
        current_team_drivers=current_team_drivers,
        current_team_ctors=current_team_ctors,
        free_transfers=free_transfers,
    )
    if penalised is None:
        raise RuntimeError(
            "Penalised optimisation was infeasible — check current team and predictions"
        )

    n = penalised["n_transfers"]
    excess = penalised["excess_transfers"]
    log.info(
        "  Recommended: %d transfer%s (free: %d, excess: %d → -%d pts penalty)",
        n,
        "s" if n != 1 else "",
        min(n, free_transfers),
        excess,
        int(penalised["transfer_cost"]),
    )
    log.info(
        "  Net predicted score: %.1f pts  (gross %.1f − %.0f penalty)",
        penalised["predicted_score"],
        penalised["gross_score"],
        penalised["transfer_cost"],
    )

    # 2) Unconstrained solve — theoretical ceiling (no current team, no penalty)
    #    This is the team you'd pick if you had unlimited free transfers.
    unconstrained = optimise_team(
        drivers,
        ctors,
        budget=budget,
        turbo_driver=turbo_driver,
    )
    if unconstrained is not None:
        log.info(
            "  Unconstrained ceiling: %.1f pts (%d transfers from current team)",
            unconstrained["gross_score"],
            # Count how many of the unconstrained team differ from current
            len(set(unconstrained["drivers"]) - set(current_team_drivers))
            + len(set(unconstrained["constructors"]) - set(current_team_ctors or [])),
        )
    else:
        log.warning("  Unconstrained solve failed — using penalised only")

    gross_delta = 0.0
    if unconstrained is not None:
        gross_delta = unconstrained["gross_score"] - penalised["gross_score"]

    if gross_delta > 0.5:
        log.info(
            "  📊 Gap: %.1f gross pts better with unlimited transfers",
            gross_delta,
        )
    else:
        log.info(
            "  ✅ Recommended team IS the global optimum — nothing left on the table"
        )

    # Attach metadata to the recommendation
    best = penalised
    best["current_team_drivers"] = current_team_drivers
    best["current_team_ctors"] = current_team_ctors
    best["unconstrained_result"] = unconstrained
    best["gross_delta"] = gross_delta

    return best


# ── Display ───────────────────────────────────────────────────────────────────


def display_team(result: dict, drivers: pd.DataFrame, ctors: pd.DataFrame):
    """Print a clean team summary to the log."""
    log.info("━━━ Optimal F1 Fantasy Team ━━━")

    # Transfer plan (if applicable)
    if result.get("current_team_drivers"):
        n = result.get("n_transfers", 0)
        ft = result.get("free_transfers", DEFAULT_FREE_TRANSFERS)
        excess = result.get("excess_transfers", 0)
        cost = result.get("transfer_cost", 0)
        log.info("── Transfer Plan ──")
        log.info(
            "  Transfers: %d  (free: %d, excess: %d → −%d pts)",
            n,
            min(n, ft),
            excess,
            int(cost),
        )
        if n == 0:
            log.info("  → Keep current team unchanged (no transfers recommended)")
        else:
            for tx in result.get("transfers_out", []):
                log.info(
                    "  ❌ OUT: %s (%s) — $%.1fM",
                    tx["name"],
                    tx["type"],
                    tx["price"],
                )
            for tx in result.get("transfers_in", []):
                log.info(
                    "  ✅ IN:  %s (%s) — $%.1fM",
                    tx["name"],
                    tx["type"],
                    tx["price"],
                )

        # Show the gap to the unconstrained optimum
        delta = result.get("gross_delta", 0)
        if delta > 0.5:
            log.info(
                "  📊 Unconstrained optimum is %.1f gross pts better",
                delta,
            )

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
    log.info("  Turbo Driver:     %s (pts doubled)", result["turbo"])
    log.info("  Total Cost:       $%.1fM", result["total_cost"])
    log.info("  Budget Remaining: $%.1fM", result["budget_remaining"])
    if result.get("transfer_cost", 0) > 0:
        log.info(
            "  Predicted Score:  %.1f pts  (%.1f gross − %.0f transfer penalty)",
            result["predicted_score"],
            result["gross_score"],
            result["transfer_cost"],
        )
    else:
        log.info("  Predicted Score:  %.1f pts", result["predicted_score"])
    log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


# ── Load Scraped Prices ───────────────────────────────────────────────────────


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
    current_team: str | None = None,
    free_transfers: int = DEFAULT_FREE_TRANSFERS,
    exclude_drivers: list[str] | None = None,
    exclude_ctors: list[str] | None = None,
    season: int | None = None,
    round_number: int | None = None,
) -> dict:
    """
    Run the full team selection pipeline.

    Args:
        budget:          Total budget in $M (default 100).
        turbo_driver:    Force a specific turbo driver code, or None to auto-pick.
        current_team:    Comma-separated string of current team
                         (e.g. "VER,NOR,LEC,RUS,PIA,McLaren,Ferrari"),
                         or None for an unconstrained (greenfield) pick.
        free_transfers:  Number of free transfers allowed (default 2).
        exclude_drivers: Driver codes to exclude from selection.
        exclude_ctors:   Constructor names to exclude from selection.
        season:          Season year (for loading round-specific predictions).
        round_number:    Round number (for loading round-specific predictions).

    Returns:
        Result dict from the optimiser, including transfer plan when applicable.
    """
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

    # Parse current team if provided
    current_team_drivers = None
    current_team_ctors = None
    if current_team:
        current_team_drivers, current_team_ctors = parse_current_team(current_team)
        log.info(
            "Current team parsed — drivers: %s, constructors: %s",
            current_team_drivers,
            current_team_ctors,
        )

    result = optimise_with_transfers(
        drivers,
        ctors,
        budget=budget,
        turbo_driver=turbo_driver,
        current_team_drivers=current_team_drivers,
        current_team_ctors=current_team_ctors,
        free_transfers=free_transfers,
    )

    display_team(result, drivers, ctors)

    # Attach the full enriched dataframes so callers (e.g. report generator)
    # don't need to re-load and re-enrich predictions a second time.
    result["all_drivers"] = drivers
    result["all_ctors"] = ctors

    log.info("━━━ Done ━━━")
    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="F1 Fantasy Team Selector — optimal team with transfer awareness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Best unconstrained team (no current team):
  python src/optimiser/team_selector.py

  # Best team reachable with up to 2 free transfers:
  python src/optimiser/team_selector.py \\
      --current-team VER,NOR,LEC,RUS,PIA,McLaren,Ferrari

  # Wildcard week — unlimited free transfers:
  python src/optimiser/team_selector.py \\
      --current-team VER,NOR,LEC,RUS,PIA,McLaren,Ferrari \\
      --free-transfers 7

  # Force a specific turbo driver:
  python src/optimiser/team_selector.py --turbo VER
        """,
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=100.0,
        help="Budget cap in $M (default: 100.0)",
    )
    parser.add_argument(
        "--turbo",
        type=str,
        default=None,
        help="Force a specific turbo driver code (e.g. VER)",
    )
    parser.add_argument(
        "--current-team",
        type=str,
        default=None,
        dest="current_team",
        help=(
            "Comma-separated current team: 5 driver codes + 2 constructor names. "
            "E.g. VER,NOR,LEC,RUS,PIA,McLaren,Ferrari. "
            "When provided, the optimiser limits changes to --free-transfers."
        ),
    )
    parser.add_argument(
        "--free-transfers",
        type=int,
        default=DEFAULT_FREE_TRANSFERS,
        dest="free_transfers",
        help=f"Number of free transfers available (default: {DEFAULT_FREE_TRANSFERS})",
    )
    parser.add_argument(
        "--exclude-drivers",
        type=str,
        default=None,
        dest="exclude_drivers",
        help="Comma-separated driver codes to exclude (e.g. BOT,STR)",
    )
    parser.add_argument(
        "--exclude-ctors",
        type=str,
        default=None,
        dest="exclude_ctors",
        help="Comma-separated constructor names to exclude",
    )
    parser.add_argument(
        "--season",
        type=int,
        default=None,
        help="Season year (loads round-specific predictions if --round also given)",
    )
    parser.add_argument(
        "--round",
        type=int,
        default=None,
        dest="round_number",
        help="Round number (loads round-specific predictions if --season also given)",
    )

    args = parser.parse_args()

    exc_d = (
        [x.strip() for x in args.exclude_drivers.split(",")]
        if args.exclude_drivers
        else None
    )
    exc_c = (
        [x.strip() for x in args.exclude_ctors.split(",")]
        if args.exclude_ctors
        else None
    )

    run(
        budget=args.budget,
        turbo_driver=args.turbo,
        current_team=args.current_team,
        free_transfers=args.free_transfers,
        exclude_drivers=exc_d,
        exclude_ctors=exc_c,
        season=args.season,
        round_number=args.round_number,
    )
