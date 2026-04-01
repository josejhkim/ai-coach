from __future__ import annotations

import pytest
from pydantic import ValidationError

from coach.model.params import InfluenceWeights, MatchupParams, PlayerParams, RallyStyleMix, ServeMix


def make_player(player_id: str, name: str) -> PlayerParams:
    return PlayerParams(
        player_id=player_id,
        name=name,
        base_srv_win=0.57,
        base_rcv_win=0.52,
        unforced_error_rate=0.18,
        return_pressure=0.52,
        clutch_point_win=0.51,
        serve_mix=ServeMix(short=0.65, flick=0.35),
        rally_style=RallyStyleMix(attack=0.5, neutral=0.3, safe=0.2),
        sample_matches=20,
    )


def test_serve_mix_must_sum_to_one() -> None:
    with pytest.raises(ValidationError):
        ServeMix(short=0.8, flick=0.3)


def test_rally_style_must_sum_to_one() -> None:
    with pytest.raises(ValidationError):
        RallyStyleMix(attack=0.5, neutral=0.4, safe=0.2)


def test_matchup_constraints_and_adjustments() -> None:
    params = MatchupParams(
        player_a=make_player("a", "Player A"),
        player_b=make_player("b", "Player B"),
        weights=InfluenceWeights(w_short=0.04, w_attack=0.06, w_safe=0.05),
    )
    eff = params.effective_probabilities()
    assert 0.0 < eff["pA_srv_win"] < 1.0
    assert 0.0 < eff["pA_rcv_win"] < 1.0

    adjusted = params.with_adjustments(
        serve_short_delta=0.05,
        attack_delta=-0.1,
        unforced_error_delta=-0.02,
        return_pressure_delta=0.03,
        clutch_delta=0.02,
    )
    assert abs(adjusted.player_a.serve_mix.short + adjusted.player_a.serve_mix.flick - 1.0) < 1e-9
    assert (
        abs(
            adjusted.player_a.rally_style.attack
            + adjusted.player_a.rally_style.neutral
            + adjusted.player_a.rally_style.safe
            - 1.0
        )
        < 1e-9
    )
    assert adjusted.player_a.unforced_error_rate < params.player_a.unforced_error_rate
    assert adjusted.player_a.return_pressure > params.player_a.return_pressure
    assert adjusted.player_a.clutch_point_win > params.player_a.clutch_point_win


def test_unforced_error_bounds_are_validated() -> None:
    with pytest.raises(ValidationError):
        PlayerParams(
            player_id="a",
            name="Player A",
            base_srv_win=0.57,
            base_rcv_win=0.52,
            unforced_error_rate=0.75,
            return_pressure=0.52,
            clutch_point_win=0.51,
            serve_mix=ServeMix(short=0.65, flick=0.35),
            rally_style=RallyStyleMix(attack=0.5, neutral=0.3, safe=0.2),
            sample_matches=20,
        )


def test_effective_probability_improves_with_lower_unforced_errors() -> None:
    base = MatchupParams(
        player_a=make_player("a", "Player A"),
        player_b=make_player("b", "Player B"),
        weights=InfluenceWeights(
            w_short=0.04,
            w_attack=0.06,
            w_safe=0.05,
            w_ue=0.08,
            w_return_pressure=0.07,
            w_clutch=0.05,
        ),
    )
    improved = base.with_adjustments(unforced_error_delta=-0.03)
    assert improved.effective_probabilities()["pA_srv_win"] > base.effective_probabilities()["pA_srv_win"]
    assert improved.effective_probabilities()["pA_rcv_win"] > base.effective_probabilities()["pA_rcv_win"]


def test_best_of_must_be_odd() -> None:
    with pytest.raises(ValidationError):
        MatchupParams(
            player_a=make_player("a", "Player A"),
            player_b=make_player("b", "Player B"),
            weights=InfluenceWeights(w_short=0.04, w_attack=0.06, w_safe=0.05),
            best_of=2,
        )


def test_reliability_scaling_damps_new_feature_edges() -> None:
    player_a = make_player("a", "Player A").model_copy(
        update={
            "short_serve_skill": 0.8,
            "long_serve_skill": 0.75,
            "rally_tolerance": 0.7,
            "net_error_rate": 0.05,
            "out_error_rate": 0.08,
            "reliability": 1.0,
        }
    )
    player_b = make_player("b", "Player B").model_copy(
        update={
            "short_serve_skill": 0.35,
            "long_serve_skill": 0.4,
            "rally_tolerance": 0.45,
            "net_error_rate": 0.2,
            "out_error_rate": 0.18,
            "reliability": 1.0,
        }
    )
    high_rel = MatchupParams(
        player_a=player_a,
        player_b=player_b,
        weights=InfluenceWeights(w_short=0.04, w_attack=0.06, w_safe=0.05, w_serve_type=0.08, w_error_profile=0.08),
    )
    low_rel = MatchupParams(
        player_a=player_a.model_copy(update={"reliability": 0.1}),
        player_b=player_b.model_copy(update={"reliability": 0.1}),
        weights=InfluenceWeights(w_short=0.04, w_attack=0.06, w_safe=0.05, w_serve_type=0.08, w_error_profile=0.08),
    )

    assert high_rel.effective_probabilities()["pA_srv_win"] > low_rel.effective_probabilities()["pA_srv_win"]
    assert high_rel.effective_probabilities()["pA_rcv_win"] > low_rel.effective_probabilities()["pA_rcv_win"]


def test_handedness_edge_is_directional_and_bounded() -> None:
    base_weights = InfluenceWeights(w_short=0.04, w_attack=0.06, w_safe=0.05, w_handedness=0.08)
    all_right = MatchupParams(
        player_a=make_player("a", "Player A").model_copy(update={"handedness_flag": 0.0}),
        player_b=make_player("b", "Player B").model_copy(update={"handedness_flag": 0.0}),
        weights=base_weights,
    )
    a_left_vs_right = MatchupParams(
        player_a=make_player("a", "Player A").model_copy(update={"handedness_flag": 1.0}),
        player_b=make_player("b", "Player B").model_copy(update={"handedness_flag": 0.0}),
        weights=base_weights,
    )

    assert a_left_vs_right.effective_probabilities()["pA_srv_win"] >= all_right.effective_probabilities()["pA_srv_win"]
    assert a_left_vs_right.effective_probabilities()["pA_rcv_win"] >= all_right.effective_probabilities()["pA_rcv_win"]
    assert 0.01 <= a_left_vs_right.effective_probabilities()["pA_srv_win"] <= 0.99
    assert 0.01 <= a_left_vs_right.effective_probabilities()["pA_rcv_win"] <= 0.99


def test_stroke_profile_edges_influence_probabilities() -> None:
    baseline = MatchupParams(
        player_a=make_player("a", "Player A").model_copy(update={"backhand_rate": 0.45, "aroundhead_rate": 0.05}),
        player_b=make_player("b", "Player B").model_copy(update={"backhand_rate": 0.45, "aroundhead_rate": 0.05}),
        weights=InfluenceWeights(w_short=0.04, w_attack=0.06, w_safe=0.05, w_backhand=0.08, w_aroundhead=0.08),
    )
    improved = MatchupParams(
        player_a=make_player("a", "Player A").model_copy(update={"backhand_rate": 0.25, "aroundhead_rate": 0.18}),
        player_b=make_player("b", "Player B").model_copy(update={"backhand_rate": 0.45, "aroundhead_rate": 0.05}),
        weights=InfluenceWeights(w_short=0.04, w_attack=0.06, w_safe=0.05, w_backhand=0.08, w_aroundhead=0.08),
    )

    assert improved.effective_probabilities()["pA_srv_win"] > baseline.effective_probabilities()["pA_srv_win"]
    assert improved.effective_probabilities()["pA_rcv_win"] > baseline.effective_probabilities()["pA_rcv_win"]
