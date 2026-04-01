from __future__ import annotations

import pandas as pd
import pytest

from coach.data.adapters.local_csv import LocalCSVAdapter
from coach.data.stats_builder import build_matchup_params, estimate_influence_weights
from coach.service import BadmintonCoachService


def test_local_stats_include_fine_grained_metrics() -> None:
    adapter = LocalCSVAdapter()
    stats = adapter.get_player_params("viktor_axelsen", window=30)

    assert 0.01 <= float(stats["unforced_error_rate"]) <= 0.6
    assert 0.01 <= float(stats["return_pressure"]) <= 0.99
    assert 0.01 <= float(stats["clutch_point_win"]) <= 0.99
    assert 0.01 <= float(stats["short_serve_skill"]) <= 0.99
    assert 0.01 <= float(stats["long_serve_skill"]) <= 0.99
    assert 0.01 <= float(stats["rally_tolerance"]) <= 0.99
    assert 0.0 <= float(stats["net_error_rate"]) <= 1.0
    assert 0.0 <= float(stats["out_error_rate"]) <= 1.0
    assert 0.0 <= float(stats["backhand_rate"]) <= 1.0
    assert 0.0 <= float(stats["aroundhead_rate"]) <= 1.0
    assert 0.0 <= float(stats["reliability"]) <= 1.0


def test_unforced_error_proxy_supports_vectorized_inputs() -> None:
    attack = pd.Series([0.56, 0.22, 0.91])
    safe = pd.Series([0.16, 0.48, 0.05])
    flick = pd.Series([0.33, 0.12, 0.40])
    points_for = pd.Series([21.0, 18.0, 30.0])
    points_against = pd.Series([17.0, 21.0, 29.0])

    vectorized = LocalCSVAdapter._estimate_unforced_error_proxy(
        attack_rate=attack,
        safe_rate=safe,
        flick_rate=flick,
        points_for=points_for,
        points_against=points_against,
    )

    expected = pd.Series(
        [
            LocalCSVAdapter._estimate_unforced_error_proxy(
                attack_rate=float(attack.iloc[i]),
                safe_rate=float(safe.iloc[i]),
                flick_rate=float(flick.iloc[i]),
                points_for=float(points_for.iloc[i]),
                points_against=float(points_against.iloc[i]),
            )
            for i in range(len(attack))
        ]
    )

    pd.testing.assert_series_equal(vectorized, expected)


@pytest.mark.parametrize(
    ("points_for", "points_against"),
    [
        (21.0, pd.Series([17.0, 21.0, 29.0])),
        (pd.Series([21.0, 18.0, 30.0]), 17.0),
    ],
)
def test_unforced_error_proxy_supports_mixed_scalar_and_series_points(
    points_for: float | pd.Series,
    points_against: float | pd.Series,
) -> None:
    attack = pd.Series([0.56, 0.22, 0.91])
    safe = pd.Series([0.16, 0.48, 0.05])
    flick = pd.Series([0.33, 0.12, 0.40])

    vectorized = LocalCSVAdapter._estimate_unforced_error_proxy(
        attack_rate=attack,
        safe_rate=safe,
        flick_rate=flick,
        points_for=points_for,
        points_against=points_against,
    )

    expected = pd.Series(
        [
            LocalCSVAdapter._estimate_unforced_error_proxy(
                attack_rate=float(attack.iloc[i]),
                safe_rate=float(safe.iloc[i]),
                flick_rate=float(flick.iloc[i]),
                points_for=float(points_for if isinstance(points_for, float) else points_for.iloc[i]),
                points_against=float(
                    points_against if isinstance(points_against, float) else points_against.iloc[i]
                ),
            )
            for i in range(len(attack))
        ]
    )

    pd.testing.assert_series_equal(vectorized, expected)


def test_strategy_candidate_generator_has_micro_steps_and_new_knobs(tmp_path) -> None:
    service = BadmintonCoachService(runs_root=tmp_path)
    baseline, _ = build_matchup_params(
        adapter=service.adapter,
        player_a_ref="Viktor Axelsen",
        player_b_ref="Kento Momota",
        window=30,
    )

    candidates = service._generate_candidates(baseline=baseline, l1_bound=0.35)
    assert candidates
    assert all("serve_short_delta" in c for c in candidates)
    assert all("attack_delta" in c for c in candidates)
    assert all("unforced_error_delta" in c for c in candidates)
    assert all("return_pressure_delta" in c for c in candidates)
    assert all("clutch_delta" in c for c in candidates)
    assert all("serve_effectiveness_delta" in c for c in candidates)
    assert all("error_profile_delta" in c for c in candidates)
    assert all("rally_tolerance_delta" in c for c in candidates)
    assert all("l1_change" in c for c in candidates)

    assert any(abs(c["serve_short_delta"]) == 0.01 for c in candidates)
    assert any(abs(c["attack_delta"]) == 0.01 for c in candidates)
    assert any(abs(c["unforced_error_delta"]) == 0.01 for c in candidates)
    assert any(abs(c["return_pressure_delta"]) == 0.01 for c in candidates)
    assert any(abs(c["clutch_delta"]) == 0.01 for c in candidates)
    assert any(abs(c["serve_effectiveness_delta"]) == 0.01 for c in candidates)
    assert any(abs(c["error_profile_delta"]) == 0.01 for c in candidates)
    assert any(abs(c["rally_tolerance_delta"]) == 0.01 for c in candidates)
    assert [c["l1_change"] for c in candidates] == sorted(c["l1_change"] for c in candidates)


def test_estimated_weights_include_calibrated_rally_tolerance_and_stroke_terms() -> None:
    adapter = LocalCSVAdapter()
    weights = estimate_influence_weights(adapter)

    assert 0.0 <= float(weights.w_rally_tolerance) <= 0.08
    assert 0.0 <= float(weights.w_backhand) <= 0.08
    assert 0.0 <= float(weights.w_aroundhead) <= 0.08
