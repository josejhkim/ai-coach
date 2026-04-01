from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts import build_real_data as brd


def _fixture_set_rows() -> list[dict[str, object]]:
    # Three rallies with known outcomes:
    # - Rally 1: A short-serve wins (terminal reason not error bucket)
    # - Rally 2: B long-serve loses due to out error
    # - Rally 3: A long-serve loses due to net error
    return [
        # Rally 1
        {
            "rally": 1,
            "ball_round": 1,
            "type": "發短球",
            "player": "A",
            "getpoint_player": None,
            "roundscore_A": 0,
            "roundscore_B": 0,
            "backhand": None,
            "aroundhead": None,
            "lose_reason": None,
            "win_reason": None,
        },
        {
            "rally": 1,
            "ball_round": 2,
            "type": "長球",
            "player": "B",
            "getpoint_player": None,
            "roundscore_A": 0,
            "roundscore_B": 0,
            "backhand": 1,
            "aroundhead": None,
            "lose_reason": None,
            "win_reason": None,
        },
        {
            "rally": 1,
            "ball_round": 3,
            "type": "殺球",
            "player": "A",
            "getpoint_player": "A",
            "roundscore_A": 1,
            "roundscore_B": 0,
            "backhand": None,
            "aroundhead": 1,
            "lose_reason": "對手落地致勝",
            "win_reason": "落地致勝",
        },
        # Rally 2
        {
            "rally": 2,
            "ball_round": 1,
            "type": "發長球",
            "player": "B",
            "getpoint_player": None,
            "roundscore_A": 1,
            "roundscore_B": 0,
            "backhand": None,
            "aroundhead": None,
            "lose_reason": None,
            "win_reason": None,
        },
        {
            "rally": 2,
            "ball_round": 2,
            "type": "推球",
            "player": "A",
            "getpoint_player": "A",
            "roundscore_A": 2,
            "roundscore_B": 0,
            "backhand": None,
            "aroundhead": None,
            "lose_reason": "出界",
            "win_reason": "對手出界",
        },
        # Rally 3
        {
            "rally": 3,
            "ball_round": 1,
            "type": "發長球",
            "player": "A",
            "getpoint_player": None,
            "roundscore_A": 2,
            "roundscore_B": 0,
            "backhand": 1,
            "aroundhead": None,
            "lose_reason": None,
            "win_reason": None,
        },
        {
            "rally": 3,
            "ball_round": 2,
            "type": "切球",
            "player": "B",
            "getpoint_player": "B",
            "roundscore_A": 2,
            "roundscore_B": 1,
            "backhand": None,
            "aroundhead": None,
            "lose_reason": "掛網",
            "win_reason": "對手掛網",
        },
    ]


def _write_fixture_dataset(tmp_path: Path) -> Path:
    set_dir = tmp_path / "set"
    match_folder = set_dir / "fixture_match"
    match_folder.mkdir(parents=True, exist_ok=True)

    match_df = pd.DataFrame(
        [
            {
                "id": 1,
                "video": "fixture_match",
                "tournament": "Fixture Open",
                "round": "Semi-finals",
                "year": 2021,
                "month": 1,
                "day": 10,
                "set": 2,
                "duration": 47,
                "winner": "Viktor AXELSEN",
                "loser": "Kento MOMOTA",
                "downcourt": 1,
                "url": "https://example.test/match",
            }
        ]
    )
    match_df.to_csv(set_dir / "match.csv", index=False)

    set_df = pd.DataFrame(_fixture_set_rows())
    set_df.to_csv(match_folder / "set1.csv", index=False)
    return set_dir


def test_process_set_csv_computes_new_derived_metrics(tmp_path: Path) -> None:
    set_path = tmp_path / "set1.csv"
    pd.DataFrame(_fixture_set_rows()).to_csv(set_path, index=False)

    out = brd.process_set_csv(set_path)

    assert out["rally_context"]["rally_count"] == 3
    assert out["rally_context"]["rally_len_total"] == 7
    assert out["rally_context"]["long_rally_count"] == 0

    a = out["counts"]["A"]
    b = out["counts"]["B"]

    assert a["short_serve_samples"] == 1
    assert a["short_serve_wins"] == 1
    assert a["long_serve_samples"] == 1
    assert a["long_serve_wins"] == 0
    assert b["short_serve_samples"] == 0
    assert b["long_serve_samples"] == 1
    assert b["long_serve_wins"] == 0

    assert a["strokes"] == 4
    assert b["strokes"] == 3
    assert a["backhand_true"] == 1
    assert b["backhand_true"] == 1
    assert a["aroundhead_true"] == 1
    assert b["aroundhead_true"] == 0

    assert a["lost_rallies"] == 1
    assert b["lost_rallies"] == 2
    assert a["net_error_lost"] == 1
    assert b["out_error_lost"] == 1


def test_build_data_maps_new_fields_without_swap(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    set_dir = _write_fixture_dataset(tmp_path)
    monkeypatch.setattr(brd.random.Random, "random", lambda self: 0.9)

    _, matches = brd.build_data(set_dir)
    row = matches.iloc[0]

    assert row["playerA_id"] == brd.make_player_id("Viktor AXELSEN")
    assert row["playerB_id"] == brd.make_player_id("Kento MOMOTA")
    assert row["tournament"] == "Fixture Open"
    assert row["round"] == "Semi-finals"
    assert row["duration_min"] == 47
    assert row["match_sets"] == 2

    assert row["avg_rally_len"] == pytest.approx(2.3333, abs=1e-4)
    assert row["long_rally_share"] == 0.0

    assert row["a_backhand_rate"] == pytest.approx(0.25, abs=1e-4)
    assert row["b_backhand_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert row["a_aroundhead_rate"] == pytest.approx(0.25, abs=1e-4)
    assert row["b_aroundhead_rate"] == 0.0

    assert row["a_net_error_lost_rate"] == 1.0
    assert row["b_net_error_lost_rate"] == 0.0
    assert row["a_out_error_lost_rate"] == 0.0
    assert row["b_out_error_lost_rate"] == 0.5

    assert row["a_short_serve_win_rate"] == 1.0
    assert row["b_short_serve_win_rate"] == 0.5
    assert row["a_long_serve_win_rate"] == 0.0
    assert row["b_long_serve_win_rate"] == 0.0
    assert row["a_short_serve_samples"] == 1
    assert row["b_short_serve_samples"] == 0
    assert row["a_long_serve_samples"] == 1
    assert row["b_long_serve_samples"] == 1


def test_build_data_maps_new_fields_with_swap(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    set_dir = _write_fixture_dataset(tmp_path)
    monkeypatch.setattr(brd.random.Random, "random", lambda self: 0.1)

    _, matches = brd.build_data(set_dir)
    row = matches.iloc[0]

    assert row["playerA_id"] == brd.make_player_id("Kento MOMOTA")
    assert row["playerB_id"] == brd.make_player_id("Viktor AXELSEN")
    assert row["winner_id"] == brd.make_player_id("Viktor AXELSEN")

    # "a_*" now maps to original side B metrics.
    assert row["a_short_serve_win_rate"] == 0.5
    assert row["a_long_serve_win_rate"] == 0.0
    assert row["a_out_error_lost_rate"] == 0.5
    assert row["a_short_serve_samples"] == 0
    assert row["a_long_serve_samples"] == 1

    # "b_*" now maps to original side A metrics.
    assert row["b_short_serve_win_rate"] == 1.0
    assert row["b_long_serve_win_rate"] == 0.0
    assert row["b_net_error_lost_rate"] == 1.0
    assert row["b_short_serve_samples"] == 1
    assert row["b_long_serve_samples"] == 1


def test_validate_rejects_invalid_new_columns(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    set_dir = _write_fixture_dataset(tmp_path)
    monkeypatch.setattr(brd.random.Random, "random", lambda self: 0.9)
    players, matches = brd.build_data(set_dir)

    # Baseline is valid.
    brd.validate(players, matches)

    bad_neg = matches.copy()
    bad_neg.loc[0, "a_short_serve_samples"] = -1
    with pytest.raises(ValueError):
        brd.validate(players, bad_neg)

    bad_nonint = matches.copy().astype({"a_long_serve_samples": "float64"})
    bad_nonint.loc[0, "a_long_serve_samples"] = 1.5
    with pytest.raises(ValueError):
        brd.validate(players, bad_nonint)

    bad_rate = matches.copy()
    bad_rate.loc[0, "a_backhand_rate"] = 1.2
    with pytest.raises(ValueError):
        brd.validate(players, bad_rate)
