#!/usr/bin/env python3
"""Build sample_players.csv and sample_matches.csv from ShuttleSet stroke-level data.

Usage:
    python scripts/build_real_data.py                         # auto-download ShuttleSet
    python scripts/build_real_data.py --shuttleset-path data/raw/CoachAI-Projects/ShuttleSet/set
"""

from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
from pathlib import Path

import pandas as pd

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.shuttleset_config import (
    CHINESE_TO_ENGLISH,
    PLAYER_COUNTRY,
    PLAYER_HANDEDNESS,
    SERVE_TYPES,
    SHOT_CLASSIFICATION,
)

RANDOM_SEED = 42
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "coach" / "data"
LONG_RALLY_THRESHOLD = 8
NET_ERROR_REASONS = {"掛網", "未過網"}
OUT_ERROR_REASONS = {"出界"}


# --------------------------------------------------------------------------- #
# Download / locate ShuttleSet
# --------------------------------------------------------------------------- #

def ensure_shuttleset(data_dir: Path) -> Path:
    """Clone CoachAI-Projects (sparse checkout) if ShuttleSet not already present."""
    set_dir = data_dir / "CoachAI-Projects" / "ShuttleSet" / "set"
    if set_dir.exists() and (set_dir / "match.csv").exists():
        return set_dir

    repo_dir = data_dir / "CoachAI-Projects"
    print(f"Cloning ShuttleSet into {repo_dir} ...")
    subprocess.run(
        ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
         "https://github.com/wywyWang/CoachAI-Projects.git", str(repo_dir)],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo_dir), "sparse-checkout", "set", "--no-cone", "ShuttleSet"],
        check=True,
    )
    if not set_dir.exists():
        raise RuntimeError(f"ShuttleSet not found after clone at {set_dir}")
    return set_dir


# --------------------------------------------------------------------------- #
# Shot type helpers
# --------------------------------------------------------------------------- #

def translate_type(chinese_type: str) -> str:
    """Convert Chinese shot type to English, raising on unknown values."""
    english = CHINESE_TO_ENGLISH.get(chinese_type)
    if english is None:
        raise ValueError(f"Unknown shot type: {chinese_type!r}")
    return english


def classify_shot(english_type: str) -> str | None:
    """Classify an English shot type into attack/neutral/safe, or None for serves/unknown."""
    if english_type in SERVE_TYPES:
        return None  # handled separately
    if english_type == "unknown":
        return None
    category = SHOT_CLASSIFICATION.get(english_type)
    if category is None:
        raise ValueError(f"Shot type {english_type!r} has no classification")
    return category


# --------------------------------------------------------------------------- #
# Per-match aggregation
# --------------------------------------------------------------------------- #

def process_set_csv(set_path: Path) -> dict:
    """Parse one set CSV and return per-player aggregated counts."""
    df = pd.read_csv(set_path)

    counts = {
        side: {
            "short_serves": 0, "long_serves": 0,
            "serve_rallies": 0, "serve_wins": 0,
            "short_serve_samples": 0, "short_serve_wins": 0,
            "long_serve_samples": 0, "long_serve_wins": 0,
            "attack": 0, "neutral": 0, "safe": 0,
            "strokes": 0,
            "backhand_true": 0,
            "aroundhead_true": 0,
            "lost_rallies": 0,
            "net_error_lost": 0,
            "out_error_lost": 0,
        }
        for side in ("A", "B")
    }
    total_points = {"A": 0, "B": 0}
    rally_context = {
        "rally_count": 0,
        "rally_len_total": 0,
        "long_rally_count": 0,
    }

    for rally_id, rally_df in df.groupby("rally"):
        rally_df = rally_df.sort_values("ball_round")
        rally_len = int(rally_df["ball_round"].max())
        rally_context["rally_count"] += 1
        rally_context["rally_len_total"] += rally_len
        if rally_len >= LONG_RALLY_THRESHOLD:
            rally_context["long_rally_count"] += 1

        # Identify server from first stroke
        first = rally_df.iloc[0]
        english_type = translate_type(str(first["type"]))
        serve_kind: str | None = None

        if english_type not in SERVE_TYPES:
            # Malformed rally — skip serve counting but still count rally shots
            server = None
        else:
            server = str(first["player"])
            counts[server]["serve_rallies"] += 1

            if english_type == "short service":
                counts[server]["short_serves"] += 1
                counts[server]["short_serve_samples"] += 1
                serve_kind = "short"
            else:
                counts[server]["long_serves"] += 1
                counts[server]["long_serve_samples"] += 1
                serve_kind = "long"

        # Determine rally winner
        point_rows = rally_df[rally_df["getpoint_player"].notna()]
        if not point_rows.empty:
            terminal = point_rows.iloc[-1]
            winner = str(terminal["getpoint_player"])
            total_points[winner] += 1
            if server is not None and winner == server:
                counts[server]["serve_wins"] += 1
                if serve_kind == "short":
                    counts[server]["short_serve_wins"] += 1
                elif serve_kind == "long":
                    counts[server]["long_serve_wins"] += 1

            if winner in {"A", "B"}:
                loser = "B" if winner == "A" else "A"
                counts[loser]["lost_rallies"] += 1
                lose_reason_raw = terminal["lose_reason"]
                lose_reason = str(lose_reason_raw) if pd.notna(lose_reason_raw) else ""
                if lose_reason in NET_ERROR_REASONS:
                    counts[loser]["net_error_lost"] += 1
                if lose_reason in OUT_ERROR_REASONS:
                    counts[loser]["out_error_lost"] += 1

        # Classify all non-serve strokes into attack/neutral/safe
        for _, row in rally_df.iterrows():
            side = str(row["player"])
            if side not in counts:
                continue

            counts[side]["strokes"] += 1
            if pd.notna(row.get("backhand")) and float(row["backhand"]) == 1.0:
                counts[side]["backhand_true"] += 1
            if pd.notna(row.get("aroundhead")) and float(row["aroundhead"]) == 1.0:
                counts[side]["aroundhead_true"] += 1

            eng = translate_type(str(row["type"]))
            cat = classify_shot(eng)
            if cat is not None:
                counts[side][cat] += 1

    # Final set score
    last_row = df.sort_values("ball_round").groupby("rally").last().iloc[-1]
    score_a = int(last_row["roundscore_A"])
    score_b = int(last_row["roundscore_B"])

    return {
        "counts": counts,
        "total_points": total_points,
        "score_a": score_a,
        "score_b": score_b,
        "rallies": {
            "count": int(rally_context["rally_count"]),
            "len_sum": float(rally_context["rally_len_total"]),
            "long_count": int(rally_context["long_rally_count"]),
        },
        "rally_context": rally_context,
    }


def process_match(match_folder: Path, num_sets: int) -> dict:
    """Aggregate all sets of a match into match-level statistics."""

    agg = {
        side: {
            "short_serves": 0, "long_serves": 0,
            "serve_rallies": 0, "serve_wins": 0,
            "short_serve_samples": 0, "short_serve_wins": 0,
            "long_serve_samples": 0, "long_serve_wins": 0,
            "attack": 0, "neutral": 0, "safe": 0,
            "strokes": 0,
            "backhand_true": 0,
            "aroundhead_true": 0,
            "lost_rallies": 0,
            "net_error_lost": 0,
            "out_error_lost": 0,
        }
        for side in ("A", "B")
    }
    total_points = {"A": 0, "B": 0}
    games_won = {"A": 0, "B": 0}
    rally_context = {
        "rally_count": 0,
        "rally_len_total": 0,
        "long_rally_count": 0,
    }

    for set_num in range(1, num_sets + 1):
        set_path = match_folder / f"set{set_num}.csv"
        if not set_path.exists():
            continue

        result = process_set_csv(set_path)

        for side in ("A", "B"):
            for key in agg[side]:
                agg[side][key] += result["counts"][side][key]
            total_points[side] += result["total_points"][side]
        for key in rally_context:
            rally_context[key] += result["rally_context"][key]

        if result["score_a"] > result["score_b"]:
            games_won["A"] += 1
        else:
            games_won["B"] += 1

    return {
        "counts": agg,
        "total_points": total_points,
        "games_won": games_won,
        "rallies": {
            "count": int(rally_context["rally_count"]),
            "len_sum": float(rally_context["rally_len_total"]),
            "long_count": int(rally_context["long_rally_count"]),
        },
        "rally_context": rally_context,
    }


def compute_rates(counts: dict) -> dict:
    """Convert raw counts into rate fields with sum-to-1 guarantees."""
    total_serves = counts["short_serves"] + counts["long_serves"]
    if total_serves > 0:
        short_rate = counts["short_serves"] / total_serves
    else:
        short_rate = 0.5
    flick_rate = round(1.0 - round(short_rate, 4), 4)
    short_rate = round(short_rate, 4)

    total_rally = counts["attack"] + counts["neutral"] + counts["safe"]
    if total_rally > 0:
        attack = counts["attack"] / total_rally
        safe = counts["safe"] / total_rally
    else:
        attack, safe = 0.33, 0.33

    attack = round(attack, 4)
    safe = round(safe, 4)
    neutral = round(1.0 - attack - safe, 4)

    return {
        "short_serve_rate": short_rate,
        "flick_serve_rate": flick_rate,
        "attack_rate": attack,
        "neutral_rate": neutral,
        "safe_rate": safe,
    }


def compute_enriched_rates(counts: dict) -> dict:
    """Compute extended tactical/context rates with deterministic defaults."""
    strokes = int(counts["strokes"])
    lost_rallies = int(counts["lost_rallies"])
    short_samples = int(counts["short_serve_samples"])
    long_samples = int(counts["long_serve_samples"])

    backhand_rate = (counts["backhand_true"] / strokes) if strokes > 0 else 0.0
    aroundhead_rate = (counts["aroundhead_true"] / strokes) if strokes > 0 else 0.0
    net_error_lost_rate = (counts["net_error_lost"] / lost_rallies) if lost_rallies > 0 else 0.0
    out_error_lost_rate = (counts["out_error_lost"] / lost_rallies) if lost_rallies > 0 else 0.0
    short_serve_win_rate = (counts["short_serve_wins"] / short_samples) if short_samples > 0 else 0.5
    long_serve_win_rate = (counts["long_serve_wins"] / long_samples) if long_samples > 0 else 0.5

    return {
        "backhand_rate": round(float(backhand_rate), 4),
        "aroundhead_rate": round(float(aroundhead_rate), 4),
        "net_error_lost_rate": round(float(net_error_lost_rate), 4),
        "out_error_lost_rate": round(float(out_error_lost_rate), 4),
        "short_serve_win_rate": round(float(short_serve_win_rate), 4),
        "long_serve_win_rate": round(float(long_serve_win_rate), 4),
        "short_serve_samples": short_samples,
        "long_serve_samples": long_samples,
    }


# --------------------------------------------------------------------------- #
# Player registry
# --------------------------------------------------------------------------- #

def make_player_id(name: str) -> str:
    """Generate slug-style ID: 'Viktor AXELSEN' -> 'viktor_axelsen'."""
    return "_".join(name.lower().split()).replace(".", "")


def build_player_registry(all_names: set[str]) -> pd.DataFrame:
    """Build sample_players.csv dataframe from unique player names."""
    rows = []
    for name in sorted(all_names):
        if name not in PLAYER_COUNTRY:
            raise ValueError(f"Player {name!r} missing from PLAYER_COUNTRY in shuttleset_config.py")
        if name not in PLAYER_HANDEDNESS:
            raise ValueError(f"Player {name!r} missing from PLAYER_HANDEDNESS in shuttleset_config.py")

        rows.append({
            "player_id": make_player_id(name),
            "name": name,
            "country": PLAYER_COUNTRY[name],
            "handedness": PLAYER_HANDEDNESS[name],
        })

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Main ETL
# --------------------------------------------------------------------------- #

def build_data(set_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Process all ShuttleSet matches and produce players + matches DataFrames."""
    match_csv = set_dir / "match.csv"
    matches_df = pd.read_csv(match_csv)

    rng = random.Random(RANDOM_SEED)
    all_player_names: set[str] = set()
    match_rows: list[dict] = []

    for _, match_row in matches_df.iterrows():
        folder_name = str(match_row["video"])
        winner = str(match_row["winner"])
        loser = str(match_row["loser"])
        num_sets = int(match_row["set"])
        date = f"{int(match_row['year']):04d}-{int(match_row['month']):02d}-{int(match_row['day']):02d}"
        tournament = str(match_row["tournament"])
        round_name = str(match_row["round"])
        duration_min = int(match_row["duration"])
        match_sets = int(match_row["set"])

        all_player_names.add(winner)
        all_player_names.add(loser)

        match_folder = set_dir / folder_name
        if not match_folder.is_dir():
            print(f"  WARNING: folder not found for {folder_name}, skipping")
            continue

        result = process_match(match_folder, num_sets)

        # ShuttleSet: A = winner, B = loser
        # Randomly swap A/B so winner_id is not always playerA_id
        swap = rng.random() < 0.5
        if swap:
            pa_name, pb_name = loser, winner
            pa_counts = result["counts"]["B"]
            pb_counts = result["counts"]["A"]
            pa_points = result["total_points"]["B"]
            pb_points = result["total_points"]["A"]
            pa_games = result["games_won"]["B"]
            pb_games = result["games_won"]["A"]
        else:
            pa_name, pb_name = winner, loser
            pa_counts = result["counts"]["A"]
            pb_counts = result["counts"]["B"]
            pa_points = result["total_points"]["A"]
            pb_points = result["total_points"]["B"]
            pa_games = result["games_won"]["A"]
            pb_games = result["games_won"]["B"]

        pa_rates = compute_rates(pa_counts)
        pb_rates = compute_rates(pb_counts)
        pa_enriched = compute_enriched_rates(pa_counts)
        pb_enriched = compute_enriched_rates(pb_counts)
        total_rallies = int(result["rally_context"]["rally_count"])
        if total_rallies > 0:
            avg_rally_len = round(float(result["rally_context"]["rally_len_total"] / total_rallies), 4)
            long_rally_share = round(float(result["rally_context"]["long_rally_count"] / total_rallies), 4)
        else:
            avg_rally_len = 0.0
            long_rally_share = 0.0

        match_rows.append({
            "date": date,
            "playerA_id": make_player_id(pa_name),
            "playerB_id": make_player_id(pb_name),
            "winner_id": make_player_id(winner),
            "a_games_won": pa_games,
            "b_games_won": pb_games,
            "a_points": pa_points,
            "b_points": pb_points,
            "a_serve_rallies": pa_counts["serve_rallies"],
            "a_serve_wins": pa_counts["serve_wins"],
            "b_serve_rallies": pb_counts["serve_rallies"],
            "b_serve_wins": pb_counts["serve_wins"],
            "a_short_serve_rate": pa_rates["short_serve_rate"],
            "a_flick_serve_rate": pa_rates["flick_serve_rate"],
            "a_attack_rate": pa_rates["attack_rate"],
            "a_neutral_rate": pa_rates["neutral_rate"],
            "a_safe_rate": pa_rates["safe_rate"],
            "b_short_serve_rate": pb_rates["short_serve_rate"],
            "b_flick_serve_rate": pb_rates["flick_serve_rate"],
            "b_attack_rate": pb_rates["attack_rate"],
            "b_neutral_rate": pb_rates["neutral_rate"],
            "b_safe_rate": pb_rates["safe_rate"],
            "tournament": tournament,
            "round": round_name,
            "duration_min": duration_min,
            "match_sets": match_sets,
            "avg_rally_len": avg_rally_len,
            "long_rally_share": long_rally_share,
            "a_backhand_rate": pa_enriched["backhand_rate"],
            "b_backhand_rate": pb_enriched["backhand_rate"],
            "a_aroundhead_rate": pa_enriched["aroundhead_rate"],
            "b_aroundhead_rate": pb_enriched["aroundhead_rate"],
            "a_net_error_lost_rate": pa_enriched["net_error_lost_rate"],
            "b_net_error_lost_rate": pb_enriched["net_error_lost_rate"],
            "a_out_error_lost_rate": pa_enriched["out_error_lost_rate"],
            "b_out_error_lost_rate": pb_enriched["out_error_lost_rate"],
            "a_short_serve_win_rate": pa_enriched["short_serve_win_rate"],
            "b_short_serve_win_rate": pb_enriched["short_serve_win_rate"],
            "a_long_serve_win_rate": pa_enriched["long_serve_win_rate"],
            "b_long_serve_win_rate": pb_enriched["long_serve_win_rate"],
            "a_short_serve_samples": pa_enriched["short_serve_samples"],
            "b_short_serve_samples": pb_enriched["short_serve_samples"],
            "a_long_serve_samples": pa_enriched["long_serve_samples"],
            "b_long_serve_samples": pb_enriched["long_serve_samples"],
        })

    players_df = build_player_registry(all_player_names)
    matches_out = pd.DataFrame(match_rows)
    matches_out = matches_out.sort_values("date").reset_index(drop=True)

    return players_df, matches_out


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

def validate(players_df: pd.DataFrame, matches_df: pd.DataFrame) -> None:
    """Run schema and constraint checks on generated data."""
    expected_player_cols = ["player_id", "name", "country", "handedness"]
    assert list(players_df.columns) == expected_player_cols, f"Player columns: {list(players_df.columns)}"
    assert players_df["player_id"].is_unique, "Duplicate player IDs"
    assert players_df["handedness"].isin(["R", "L"]).all(), "Invalid handedness values"

    expected_match_cols = [
        "date", "playerA_id", "playerB_id", "winner_id",
        "a_games_won", "b_games_won", "a_points", "b_points",
        "a_serve_rallies", "a_serve_wins", "b_serve_rallies", "b_serve_wins",
        "a_short_serve_rate", "a_flick_serve_rate",
        "a_attack_rate", "a_neutral_rate", "a_safe_rate",
        "b_short_serve_rate", "b_flick_serve_rate",
        "b_attack_rate", "b_neutral_rate", "b_safe_rate",
        "tournament", "round", "duration_min", "match_sets",
        "avg_rally_len", "long_rally_share",
        "a_backhand_rate", "b_backhand_rate",
        "a_aroundhead_rate", "b_aroundhead_rate",
        "a_net_error_lost_rate", "b_net_error_lost_rate",
        "a_out_error_lost_rate", "b_out_error_lost_rate",
        "a_short_serve_win_rate", "b_short_serve_win_rate",
        "a_long_serve_win_rate", "b_long_serve_win_rate",
        "a_short_serve_samples", "b_short_serve_samples",
        "a_long_serve_samples", "b_long_serve_samples",
    ]
    assert list(matches_df.columns) == expected_match_cols, f"Match columns: {list(matches_df.columns)}"

    player_ids = set(players_df["player_id"])
    errors = []

    for idx, row in matches_df.iterrows():
        # FK integrity
        if row["playerA_id"] not in player_ids:
            errors.append(f"Row {idx}: playerA_id {row['playerA_id']!r} not in players")
        if row["playerB_id"] not in player_ids:
            errors.append(f"Row {idx}: playerB_id {row['playerB_id']!r} not in players")
        if row["winner_id"] not in (row["playerA_id"], row["playerB_id"]):
            errors.append(f"Row {idx}: winner_id {row['winner_id']!r} not playerA or playerB")

        # Serve rates sum to 1.0
        a_srv_sum = row["a_short_serve_rate"] + row["a_flick_serve_rate"]
        b_srv_sum = row["b_short_serve_rate"] + row["b_flick_serve_rate"]
        if abs(a_srv_sum - 1.0) > 1e-3:
            errors.append(f"Row {idx}: a serve rates sum to {a_srv_sum}")
        if abs(b_srv_sum - 1.0) > 1e-3:
            errors.append(f"Row {idx}: b serve rates sum to {b_srv_sum}")

        # Rally style rates sum to 1.0
        a_rally_sum = row["a_attack_rate"] + row["a_neutral_rate"] + row["a_safe_rate"]
        b_rally_sum = row["b_attack_rate"] + row["b_neutral_rate"] + row["b_safe_rate"]
        if abs(a_rally_sum - 1.0) > 1e-3:
            errors.append(f"Row {idx}: a rally rates sum to {a_rally_sum}")
        if abs(b_rally_sum - 1.0) > 1e-3:
            errors.append(f"Row {idx}: b rally rates sum to {b_rally_sum}")

        # Rate bounds
        rate_cols = [
            "a_short_serve_rate", "a_flick_serve_rate",
            "a_attack_rate", "a_neutral_rate", "a_safe_rate",
            "b_short_serve_rate", "b_flick_serve_rate",
            "b_attack_rate", "b_neutral_rate", "b_safe_rate",
            "avg_rally_len", "long_rally_share",
            "a_backhand_rate", "b_backhand_rate",
            "a_aroundhead_rate", "b_aroundhead_rate",
            "a_net_error_lost_rate", "b_net_error_lost_rate",
            "a_out_error_lost_rate", "b_out_error_lost_rate",
            "a_short_serve_win_rate", "b_short_serve_win_rate",
            "a_long_serve_win_rate", "b_long_serve_win_rate",
        ]
        for col in rate_cols:
            if col == "avg_rally_len":
                if row[col] < 0.0:
                    errors.append(f"Row {idx}: {col}={row[col]} must be >= 0")
            elif not (0.0 <= row[col] <= 1.0):
                errors.append(f"Row {idx}: {col}={row[col]} out of [0, 1]")

        sample_cols = [
            "a_short_serve_samples",
            "b_short_serve_samples",
            "a_long_serve_samples",
            "b_long_serve_samples",
        ]
        for col in sample_cols:
            value = float(row[col])
            if value < 0:
                errors.append(f"Row {idx}: {col}={row[col]} must be >= 0")
            if not value.is_integer():
                errors.append(f"Row {idx}: {col}={row[col]} must be an integer")

        # Serve wins <= serve rallies
        if row["a_serve_wins"] > row["a_serve_rallies"]:
            errors.append(f"Row {idx}: a_serve_wins > a_serve_rallies")
        if row["b_serve_wins"] > row["b_serve_rallies"]:
            errors.append(f"Row {idx}: b_serve_wins > b_serve_rallies")

        if int(row["match_sets"]) < 1:
            errors.append(f"Row {idx}: match_sets must be >= 1")
        if int(row["duration_min"]) <= 0:
            errors.append(f"Row {idx}: duration_min must be > 0")

    if errors:
        for e in errors:
            print(f"  VALIDATION ERROR: {e}", file=sys.stderr)
        raise ValueError(f"{len(errors)} validation errors found")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="Build real badminton data from ShuttleSet")
    parser.add_argument(
        "--shuttleset-path",
        default=None,
        help="Path to ShuttleSet/set/ directory (auto-downloads if not provided)",
    )
    args = parser.parse_args()

    if args.shuttleset_path:
        set_dir = Path(args.shuttleset_path).expanduser().resolve()
    else:
        raw_dir = Path(__file__).resolve().parents[1] / "data" / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        set_dir = ensure_shuttleset(raw_dir)

    print(f"Reading ShuttleSet from: {set_dir}")
    players_df, matches_df = build_data(set_dir)

    print(f"Players: {len(players_df)}")
    print(f"Matches: {len(matches_df)}")

    validate(players_df, matches_df)
    print("Validation passed.")

    players_path = OUTPUT_DIR / "sample_players.csv"
    matches_path = OUTPUT_DIR / "sample_matches.csv"

    players_df.to_csv(players_path, index=False)
    matches_df.to_csv(matches_path, index=False)

    print(f"Wrote {players_path}")
    print(f"Wrote {matches_path}")


if __name__ == "__main__":
    main()
