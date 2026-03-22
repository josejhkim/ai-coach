from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from coach.utils import clamp


class ServeMix(BaseModel):
    model_config = ConfigDict(extra="forbid")

    short: float = Field(..., ge=0.0, le=1.0)
    flick: float = Field(..., ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_sum(self) -> "ServeMix":
        total = self.short + self.flick
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Serve mix must sum to 1.0; got {total:.6f}")
        return self


class RallyStyleMix(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attack: float = Field(..., ge=0.0, le=1.0)
    neutral: float = Field(..., ge=0.0, le=1.0)
    safe: float = Field(..., ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_sum(self) -> "RallyStyleMix":
        total = self.attack + self.neutral + self.safe
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Rally style mix must sum to 1.0; got {total:.6f}")
        return self


class InfluenceWeights(BaseModel):
    model_config = ConfigDict(extra="forbid")

    w_short: float = Field(..., gt=0.0, le=0.3)
    w_attack: float = Field(..., gt=0.0, le=0.3)
    w_safe: float = Field(..., gt=0.0, le=0.3)
    w_ue: float = Field(default=0.08, gt=0.0, le=0.3)
    w_return_pressure: float = Field(default=0.07, gt=0.0, le=0.3)
    w_clutch: float = Field(default=0.05, gt=0.0, le=0.2)


class PlayerParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    player_id: str
    name: str
    base_srv_win: float = Field(..., ge=0.01, le=0.99)
    base_rcv_win: float = Field(..., ge=0.01, le=0.99)
    unforced_error_rate: float = Field(default=0.18, ge=0.01, le=0.6)
    return_pressure: float = Field(default=0.5, ge=0.01, le=0.99)
    clutch_point_win: float = Field(default=0.5, ge=0.01, le=0.99)
    serve_mix: ServeMix
    rally_style: RallyStyleMix
    sample_matches: int = Field(default=0, ge=0)

    @field_validator("player_id", "name")
    @classmethod
    def not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("String fields must not be empty")
        return value


class MatchupParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    player_a: PlayerParams
    player_b: PlayerParams
    weights: InfluenceWeights
    target: int = Field(default=21, ge=11, le=30)
    cap: int = Field(default=30, ge=21, le=50)
    best_of: int = Field(default=3, ge=1, le=7)

    @model_validator(mode="after")
    def validate_game_constraints(self) -> "MatchupParams":
        if self.best_of % 2 == 0:
            raise ValueError("best_of must be odd")
        if self.cap < self.target:
            raise ValueError("cap must be >= target")
        return self

    def _style_delta(self) -> float:
        a = self.player_a
        b = self.player_b
        return (
            self.weights.w_short * (a.serve_mix.short - b.serve_mix.short)
            + self.weights.w_attack * (a.rally_style.attack - b.rally_style.attack)
            - self.weights.w_safe * (b.rally_style.safe - a.rally_style.safe)
        )

    def _micro_edges(self) -> dict[str, float]:
        a = self.player_a
        b = self.player_b
        return {
            # Lower unforced-error rate is better for player A.
            "ue_edge": b.unforced_error_rate - a.unforced_error_rate,
            "return_edge": a.return_pressure - b.return_pressure,
            "clutch_edge": a.clutch_point_win - b.clutch_point_win,
        }

    def effective_probabilities(self) -> dict[str, float]:
        a = self.player_a
        b = self.player_b
        style_delta = self._style_delta()
        edges = self._micro_edges()

        serve_delta = (
            style_delta
            + self.weights.w_ue * edges["ue_edge"]
            + 0.55 * self.weights.w_return_pressure * edges["return_edge"]
            + self.weights.w_clutch * edges["clutch_edge"]
        )
        receive_style_delta = (
            0.45 * self.weights.w_short * (a.serve_mix.short - b.serve_mix.short)
            + 0.75 * self.weights.w_attack * (a.rally_style.attack - b.rally_style.attack)
            - 0.75 * self.weights.w_safe * (b.rally_style.safe - a.rally_style.safe)
        )
        receive_delta = (
            receive_style_delta
            + 0.65 * self.weights.w_ue * edges["ue_edge"]
            + 0.95 * self.weights.w_return_pressure * edges["return_edge"]
            + 0.70 * self.weights.w_clutch * edges["clutch_edge"]
        )

        p_a_srv = clamp(a.base_srv_win + serve_delta)
        p_a_rcv = clamp(a.base_rcv_win + receive_delta)

        return {
            "pA_srv_win": p_a_srv,
            "pA_rcv_win": p_a_rcv,
            "pB_srv_win": clamp(1.0 - p_a_rcv),
            "pB_rcv_win": clamp(1.0 - p_a_srv),
        }

    def with_adjustments(
        self,
        serve_short_delta: float = 0.0,
        attack_delta: float = 0.0,
        unforced_error_delta: float = 0.0,
        return_pressure_delta: float = 0.0,
        clutch_delta: float = 0.0,
    ) -> "MatchupParams":
        a = self.player_a

        short = clamp(a.serve_mix.short + serve_short_delta, 0.01, 0.99)
        serve_mix = ServeMix(short=short, flick=1.0 - short)

        attack = clamp(a.rally_style.attack + attack_delta, 0.01, 0.98)
        remain_old = max(a.rally_style.neutral + a.rally_style.safe, 1e-6)
        neutral = (a.rally_style.neutral / remain_old) * (1.0 - attack)
        safe = (a.rally_style.safe / remain_old) * (1.0 - attack)
        rally_style = RallyStyleMix(attack=attack, neutral=neutral, safe=safe)

        new_a = a.model_copy(
            update={
                "serve_mix": serve_mix,
                "rally_style": rally_style,
                "unforced_error_rate": clamp(a.unforced_error_rate + unforced_error_delta, 0.01, 0.6),
                "return_pressure": clamp(a.return_pressure + return_pressure_delta, 0.01, 0.99),
                "clutch_point_win": clamp(a.clutch_point_win + clutch_delta, 0.01, 0.99),
            }
        )
        return self.model_copy(update={"player_a": new_a})

    def l1_change_from(self, baseline: "MatchupParams") -> float:
        a_now = self.player_a
        a_base = baseline.player_a
        return (
            abs(a_now.serve_mix.short - a_base.serve_mix.short)
            + abs(a_now.serve_mix.flick - a_base.serve_mix.flick)
            + abs(a_now.rally_style.attack - a_base.rally_style.attack)
            + abs(a_now.rally_style.neutral - a_base.rally_style.neutral)
            + abs(a_now.rally_style.safe - a_base.rally_style.safe)
            + abs(a_now.unforced_error_rate - a_base.unforced_error_rate)
            + abs(a_now.return_pressure - a_base.return_pressure)
            + abs(a_now.clutch_point_win - a_base.clutch_point_win)
        )

    def to_template_context(self) -> dict[str, Any]:
        eff = self.effective_probabilities()
        scale = 10000

        p_a_srv_w = int(round(eff["pA_srv_win"] * scale))
        p_a_rcv_w = int(round(eff["pA_rcv_win"] * scale))

        context = {
            "target": self.target,
            "cap": self.cap,
            "best_of": self.best_of,
            "games_to_win": (self.best_of // 2) + 1,
            "pA_srv_win": f"{eff['pA_srv_win']:.6f}",
            "pA_rcv_win": f"{eff['pA_rcv_win']:.6f}",
            "pA_srv_win_w": p_a_srv_w,
            "pA_srv_lose_w": scale - p_a_srv_w,
            "pA_rcv_win_w": p_a_rcv_w,
            "pA_rcv_lose_w": scale - p_a_rcv_w,
            "baseA_srv_win": f"{self.player_a.base_srv_win:.6f}",
            "baseA_rcv_win": f"{self.player_a.base_rcv_win:.6f}",
            "baseB_srv_win": f"{self.player_b.base_srv_win:.6f}",
            "baseB_rcv_win": f"{self.player_b.base_rcv_win:.6f}",
            "unforced_error_A": f"{self.player_a.unforced_error_rate:.6f}",
            "unforced_error_B": f"{self.player_b.unforced_error_rate:.6f}",
            "ue_rate_A": f"{self.player_a.unforced_error_rate:.6f}",
            "ue_rate_B": f"{self.player_b.unforced_error_rate:.6f}",
            "return_pressure_A": f"{self.player_a.return_pressure:.6f}",
            "return_pressure_B": f"{self.player_b.return_pressure:.6f}",
            "clutch_A": f"{self.player_a.clutch_point_win:.6f}",
            "clutch_B": f"{self.player_b.clutch_point_win:.6f}",
            "serve_mix_A_short": f"{self.player_a.serve_mix.short:.6f}",
            "serve_mix_A_flick": f"{self.player_a.serve_mix.flick:.6f}",
            "serve_mix_B_short": f"{self.player_b.serve_mix.short:.6f}",
            "serve_mix_B_flick": f"{self.player_b.serve_mix.flick:.6f}",
            "rally_style_A_attack": f"{self.player_a.rally_style.attack:.6f}",
            "rally_style_A_neutral": f"{self.player_a.rally_style.neutral:.6f}",
            "rally_style_A_safe": f"{self.player_a.rally_style.safe:.6f}",
            "rally_style_B_attack": f"{self.player_b.rally_style.attack:.6f}",
            "rally_style_B_neutral": f"{self.player_b.rally_style.neutral:.6f}",
            "rally_style_B_safe": f"{self.player_b.rally_style.safe:.6f}",
            "w_short": f"{self.weights.w_short:.6f}",
            "w_attack": f"{self.weights.w_attack:.6f}",
            "w_safe": f"{self.weights.w_safe:.6f}",
            "w_ue": f"{self.weights.w_ue:.6f}",
            "w_return_pressure": f"{self.weights.w_return_pressure:.6f}",
            "w_clutch": f"{self.weights.w_clutch:.6f}",
            "playerA_name": self.player_a.name,
            "playerB_name": self.player_b.name,
        }
        return context
