from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from coach.data.adapters.local_csv import LocalCSVAdapter, PlayerRecord
from coach.model.params import InfluenceWeights, MatchupParams, PlayerParams, RallyStyleMix, ServeMix
from coach.utils import clamp


@dataclass(frozen=True)
class MatchupStats:
    player_a: PlayerRecord
    player_b: PlayerRecord
    player_a_stats: dict[str, Any]
    player_b_stats: dict[str, Any]
    head_to_head: dict[str, Any]
    weights: InfluenceWeights


def _resolve_player(adapter: LocalCSVAdapter, player_ref: str) -> PlayerRecord:
    by_id = adapter.players_df[adapter.players_df["player_id"] == player_ref]
    if not by_id.empty:
        row = by_id.iloc[0]
        return PlayerRecord(
            player_id=str(row["player_id"]),
            name=str(row["name"]),
            country=str(row.get("country", "")) or None,
            handedness=str(row.get("handedness", "")) or None,
        )
    return adapter.resolve_player(player_ref)


def estimate_influence_weights(adapter: LocalCSVAdapter) -> InfluenceWeights:
    df = adapter.matches_df.copy()
    if len(df) < 10:
        return InfluenceWeights(
            w_short=0.04,
            w_attack=0.06,
            w_safe=0.05,
            w_ue=0.08,
            w_return_pressure=0.07,
            w_clutch=0.05,
            w_serve_type=0.03,
            w_rally_tolerance=0.02,
            w_error_profile=0.03,
            w_handedness=0.01,
            w_backhand=0.01,
            w_aroundhead=0.01,
        )

    feature_rows: list[list[float]] = []
    targets: list[float] = []
    stats_cache: dict[tuple[str, str], dict[str, Any] | None] = {}

    def get_historical_stats(player_id: str, cutoff: str) -> dict[str, Any] | None:
        key = (player_id, cutoff)
        if key not in stats_cache:
            try:
                stats_cache[key] = adapter.get_player_params(player_id=player_id, window=30, as_of_date=cutoff)
            except ValueError:
                stats_cache[key] = None
        return stats_cache[key]

    for row in df.sort_values("date").itertuples(index=False):
        cutoff = (pd.Timestamp(row.date) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        a_stats = get_historical_stats(str(row.playerA_id), cutoff)
        b_stats = get_historical_stats(str(row.playerB_id), cutoff)
        if a_stats is None or b_stats is None:
            continue

        total_points = max(float(row.a_points + row.b_points), 1.0)
        feature_rows.append([
            float(a_stats["serve_mix"]["short"]) - float(b_stats["serve_mix"]["short"]),
            float(a_stats["rally_style"]["attack"]) - float(b_stats["rally_style"]["attack"]),
            float(a_stats["rally_style"]["safe"]) - float(b_stats["rally_style"]["safe"]),
            float(b_stats["unforced_error_rate"]) - float(a_stats["unforced_error_rate"]),
            float(a_stats["return_pressure"]) - float(b_stats["return_pressure"]),
            float(a_stats["clutch_point_win"]) - float(b_stats["clutch_point_win"]),
            0.5 * (float(a_stats["short_serve_skill"]) - float(b_stats["short_serve_skill"]))
            + 0.5 * (float(a_stats["long_serve_skill"]) - float(b_stats["long_serve_skill"])),
            float(a_stats["rally_tolerance"]) - float(b_stats["rally_tolerance"]),
            0.5
            * (
                (float(b_stats["net_error_rate"]) - float(a_stats["net_error_rate"]))
                + (float(b_stats["out_error_rate"]) - float(a_stats["out_error_rate"]))
            ),
            float(a_stats["handedness_flag"]) - float(b_stats["handedness_flag"]),
            float(b_stats["backhand_rate"]) - float(a_stats["backhand_rate"]),
            float(a_stats["aroundhead_rate"]) - float(b_stats["aroundhead_rate"]),
        ])
        targets.append((float(row.a_points) / total_points) - 0.5)

    if len(feature_rows) < 10:
        return InfluenceWeights(
            w_short=0.04,
            w_attack=0.06,
            w_safe=0.05,
            w_ue=0.08,
            w_return_pressure=0.07,
            w_clutch=0.05,
            w_serve_type=0.03,
            w_rally_tolerance=0.02,
            w_error_profile=0.03,
            w_handedness=0.01,
            w_backhand=0.01,
            w_aroundhead=0.01,
        )

    X = np.column_stack([
        np.ones(len(feature_rows), dtype=float),
        np.asarray(feature_rows, dtype=float),
    ])
    y = np.asarray(targets, dtype=float)

    # Ridge regression stabilizes small-sample estimates and avoids aggressive weights.
    ridge_lambda = 2.5
    xtx = X.T @ X
    reg = np.eye(xtx.shape[0], dtype=float) * ridge_lambda
    reg[0, 0] = 0.0  # keep intercept unpenalized
    beta = np.linalg.solve(xtx + reg, X.T @ y)

    w_short = float(np.clip(abs(beta[1]), 0.01, 0.2))
    w_attack = float(np.clip(abs(beta[2]), 0.01, 0.2))
    w_safe = float(np.clip(abs(beta[3]), 0.01, 0.2))
    w_ue = float(np.clip(abs(beta[4]), 0.01, 0.2))
    w_return_pressure = float(np.clip(abs(beta[5]), 0.01, 0.2))
    w_clutch = float(np.clip(abs(beta[6]), 0.01, 0.12))
    w_serve_type = float(np.clip(abs(beta[7]), 0.0, 0.08))
    w_rally_tolerance = float(np.clip(abs(beta[8]), 0.0, 0.08))
    w_error_profile = float(np.clip(abs(beta[9]), 0.0, 0.08))
    w_handedness = float(np.clip(abs(beta[10]), 0.0, 0.08))
    w_backhand = float(np.clip(abs(beta[11]), 0.0, 0.08))
    w_aroundhead = float(np.clip(abs(beta[12]), 0.0, 0.08))

    return InfluenceWeights(
        w_short=w_short,
        w_attack=w_attack,
        w_safe=w_safe,
        w_ue=w_ue,
        w_return_pressure=w_return_pressure,
        w_clutch=w_clutch,
        w_serve_type=w_serve_type,
        w_rally_tolerance=w_rally_tolerance,
        w_error_profile=w_error_profile,
        w_handedness=w_handedness,
        w_backhand=w_backhand,
        w_aroundhead=w_aroundhead,
    )


def _build_player_params(stats: dict[str, Any], sample_matches: int) -> PlayerParams:
    serve_mix = ServeMix(short=float(stats["serve_mix"]["short"]), flick=float(stats["serve_mix"]["flick"]))
    rally_style = RallyStyleMix(
        attack=float(stats["rally_style"]["attack"]),
        neutral=float(stats["rally_style"]["neutral"]),
        safe=float(stats["rally_style"]["safe"]),
    )

    return PlayerParams(
        player_id=str(stats["player_id"]),
        name=str(stats["name"]),
        base_srv_win=clamp(float(stats["base_srv_win"])),
        base_rcv_win=clamp(float(stats["base_rcv_win"])),
        unforced_error_rate=clamp(float(stats["unforced_error_rate"]), 0.01, 0.6),
        return_pressure=clamp(float(stats["return_pressure"]), 0.01, 0.99),
        clutch_point_win=clamp(float(stats["clutch_point_win"]), 0.01, 0.99),
        short_serve_skill=clamp(float(stats.get("short_serve_skill", 0.5)), 0.01, 0.99),
        long_serve_skill=clamp(float(stats.get("long_serve_skill", 0.5)), 0.01, 0.99),
        rally_tolerance=clamp(float(stats.get("rally_tolerance", 0.5)), 0.01, 0.99),
        net_error_rate=clamp(float(stats.get("net_error_rate", 0.0)), 0.0, 1.0),
        out_error_rate=clamp(float(stats.get("out_error_rate", 0.0)), 0.0, 1.0),
        backhand_rate=clamp(float(stats.get("backhand_rate", 0.0)), 0.0, 1.0),
        aroundhead_rate=clamp(float(stats.get("aroundhead_rate", 0.0)), 0.0, 1.0),
        handedness_flag=clamp(float(stats.get("handedness_flag", 0.0)), 0.0, 1.0),
        reliability=clamp(float(stats.get("reliability", 1.0)), 0.0, 1.0),
        serve_mix=serve_mix,
        rally_style=rally_style,
        sample_matches=sample_matches,
    )


def build_matchup_params(
    adapter: LocalCSVAdapter,
    player_a_ref: str,
    player_b_ref: str,
    window: int = 30,
    as_of_date: str | None = None,
) -> tuple[MatchupParams, MatchupStats]:
    player_a = _resolve_player(adapter, player_a_ref)
    player_b = _resolve_player(adapter, player_b_ref)

    if player_a.player_id == player_b.player_id:
        raise ValueError("Player A and Player B must be different players.")

    a_stats = adapter.get_player_params(player_a.player_id, window=window, as_of_date=as_of_date)
    b_stats = adapter.get_player_params(player_b.player_id, window=window, as_of_date=as_of_date)
    h2h = adapter.get_head_to_head(player_a.player_id, player_b.player_id, window=window, as_of_date=as_of_date)
    weights = estimate_influence_weights(adapter)

    blend = min(0.35, h2h["matches"] / (h2h["matches"] + 12.0)) if h2h["matches"] > 0 else 0.0

    a_stats["base_srv_win"] = (1.0 - blend) * a_stats["base_srv_win"] + blend * h2h["a_srv_win"]
    a_stats["base_rcv_win"] = (1.0 - blend) * a_stats["base_rcv_win"] + blend * h2h["a_rcv_win"]

    player_a_params = _build_player_params(a_stats, sample_matches=int(a_stats["matches"]))
    player_b_params = _build_player_params(b_stats, sample_matches=int(b_stats["matches"]))

    matchup = MatchupParams(player_a=player_a_params, player_b=player_b_params, weights=weights)
    stats = MatchupStats(
        player_a=player_a,
        player_b=player_b,
        player_a_stats=a_stats,
        player_b_stats=b_stats,
        head_to_head=h2h,
        weights=weights,
    )
    return matchup, stats
