from __future__ import annotations

import difflib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class PlayerRecord:
    player_id: str
    name: str
    country: str | None = None
    handedness: str | None = None


class LocalCSVAdapter:
    """Local CSV-backed stats adapter used by tools and CLI."""

    def __init__(
        self,
        players_path: str | Path | None = None,
        matches_path: str | Path | None = None,
        laplace_alpha: float = 2.0,
    ) -> None:
        base = Path(__file__).resolve().parents[1]
        self.players_path = Path(players_path) if players_path else base / "sample_players.csv"
        self.matches_path = Path(matches_path) if matches_path else base / "sample_matches.csv"
        self.laplace_alpha = laplace_alpha

        self._players_df: pd.DataFrame | None = None
        self._matches_df: pd.DataFrame | None = None

    @property
    def players_df(self) -> pd.DataFrame:
        if self._players_df is None:
            self._players_df = pd.read_csv(self.players_path)
        return self._players_df

    @property
    def matches_df(self) -> pd.DataFrame:
        if self._matches_df is None:
            df = pd.read_csv(self.matches_path)
            df["date"] = pd.to_datetime(df["date"], utc=False)
            df = df.sort_values("date")
            self._matches_df = df.reset_index(drop=True)
        return self._matches_df

    @staticmethod
    def _normalize_name(name: str) -> str:
        return " ".join(name.lower().strip().split())

    def resolve_player(self, name: str) -> PlayerRecord:
        normalized = self._normalize_name(name)
        players = self.players_df.copy()
        players["_norm"] = players["name"].map(self._normalize_name)

        exact = players[players["_norm"] == normalized]
        if not exact.empty:
            row = exact.iloc[0]
            return PlayerRecord(
                player_id=str(row["player_id"]),
                name=str(row["name"]),
                country=str(row.get("country", "")) or None,
                handedness=str(row.get("handedness", "")) or None,
            )

        contains = players[players["_norm"].str.contains(normalized, regex=False)]
        if len(contains) == 1:
            row = contains.iloc[0]
            return PlayerRecord(
                player_id=str(row["player_id"]),
                name=str(row["name"]),
                country=str(row.get("country", "")) or None,
                handedness=str(row.get("handedness", "")) or None,
            )

        candidates = players["name"].tolist()
        close = difflib.get_close_matches(name, candidates, n=3, cutoff=0.55)
        if close:
            raise ValueError(f"Player '{name}' not found. Did you mean: {', '.join(close)}?")
        raise ValueError(f"Player '{name}' not found in local player table.")

    def _window_filter(self, window: int, as_of_date: str | None = None) -> pd.DataFrame:
        df = self.matches_df
        if as_of_date:
            cutoff = pd.to_datetime(as_of_date)
            df = df[df["date"] <= cutoff]
        return df.tail(max(window * 4, window))

    def get_player_matches(self, player_id: str, window: int = 30, as_of_date: str | None = None) -> pd.DataFrame:
        df = self._window_filter(window=window, as_of_date=as_of_date)
        mask = (df["playerA_id"] == player_id) | (df["playerB_id"] == player_id)
        player_df = df.loc[mask].copy().sort_values("date")
        if player_df.empty:
            raise ValueError(f"No matches found for player_id='{player_id}'.")
        return player_df.tail(window)

    def _perspective_frame(self, df: pd.DataFrame, player_id: str) -> pd.DataFrame:
        records: list[dict[str, Any]] = []
        for row in df.itertuples(index=False):
            if row.playerA_id == player_id:
                opp_id = row.playerB_id
                serve_rallies = int(row.a_serve_rallies)
                serve_wins = int(row.a_serve_wins)
                receive_rallies = int(row.b_serve_rallies)
                receive_wins = int(row.b_serve_rallies - row.b_serve_wins)
                short_rate = float(row.a_short_serve_rate)
                flick_rate = float(row.a_flick_serve_rate)
                attack_rate = float(row.a_attack_rate)
                neutral_rate = float(row.a_neutral_rate)
                safe_rate = float(row.a_safe_rate)
                points_for = int(row.a_points)
                points_against = int(row.b_points)
                won = row.winner_id == row.playerA_id
            else:
                opp_id = row.playerA_id
                serve_rallies = int(row.b_serve_rallies)
                serve_wins = int(row.b_serve_wins)
                receive_rallies = int(row.a_serve_rallies)
                receive_wins = int(row.a_serve_rallies - row.a_serve_wins)
                short_rate = float(row.b_short_serve_rate)
                flick_rate = float(row.b_flick_serve_rate)
                attack_rate = float(row.b_attack_rate)
                neutral_rate = float(row.b_neutral_rate)
                safe_rate = float(row.b_safe_rate)
                points_for = int(row.b_points)
                points_against = int(row.a_points)
                won = row.winner_id == row.playerB_id

            records.append(
                {
                    "date": row.date,
                    "player_id": player_id,
                    "opponent_id": opp_id,
                    "serve_rallies": serve_rallies,
                    "serve_wins": serve_wins,
                    "receive_rallies": receive_rallies,
                    "receive_wins": receive_wins,
                    "short_rate": short_rate,
                    "flick_rate": flick_rate,
                    "attack_rate": attack_rate,
                    "neutral_rate": neutral_rate,
                    "safe_rate": safe_rate,
                    "points_for": points_for,
                    "points_against": points_against,
                    "won": int(won),
                }
            )
        return pd.DataFrame.from_records(records)

    def _smooth_probability(self, wins: float, trials: float, alpha: float | None = None) -> float:
        alpha = self.laplace_alpha if alpha is None else alpha
        return float((wins + alpha) / (trials + 2.0 * alpha))

    @staticmethod
    def _estimate_unforced_error_proxy(
        *,
        attack_rate: float | pd.Series,
        safe_rate: float | pd.Series,
        flick_rate: float | pd.Series,
        points_for: float | pd.Series,
        points_against: float | pd.Series,
    ) -> float | pd.Series:
        total = points_for + points_against
        total = total.clip(lower=1.0) if isinstance(total, pd.Series) else max(total, 1.0)
        point_loss = points_against / total
        proxy = (
            0.08
            + 0.22 * attack_rate
            + 0.08 * flick_rate
            + 0.11 * point_loss
            - 0.09 * safe_rate
        )
        if isinstance(proxy, pd.Series):
            return proxy.clip(lower=0.01, upper=0.6)
        return float(min(0.6, max(0.01, proxy)))

    def get_player_params(
        self,
        player_id: str,
        window: int = 30,
        as_of_date: str | None = None,
    ) -> dict[str, Any]:
        raw = self.get_player_matches(player_id=player_id, window=window, as_of_date=as_of_date)
        perspective = self._perspective_frame(raw, player_id=player_id)

        serve_trials = float(perspective["serve_rallies"].sum())
        serve_wins = float(perspective["serve_wins"].sum())
        receive_trials = float(perspective["receive_rallies"].sum())
        receive_wins = float(perspective["receive_wins"].sum())

        base_srv = self._smooth_probability(serve_wins, serve_trials)
        base_rcv = self._smooth_probability(receive_wins, receive_trials)

        serve_weight = perspective["serve_rallies"].clip(lower=1)
        rally_weight = (perspective["points_for"] + perspective["points_against"]).clip(lower=1)

        short = float((perspective["short_rate"] * serve_weight).sum() / serve_weight.sum())
        attack = float((perspective["attack_rate"] * rally_weight).sum() / rally_weight.sum())
        safe = float((perspective["safe_rate"] * rally_weight).sum() / rally_weight.sum())

        # Dirichlet-style smoothing on mix vectors.
        alpha_mix = 0.02
        short = (short + alpha_mix) / (1.0 + 2.0 * alpha_mix)
        flick = 1.0 - short

        attack = max(0.05, min(0.9, attack))
        safe = max(0.05, min(0.9, safe))
        neutral = max(0.05, 1.0 - attack - safe)
        total = attack + neutral + safe
        attack, neutral, safe = attack / total, neutral / total, safe / total

        wins = int(perspective["won"].sum())
        matches = int(len(perspective))
        points_for_total = float(perspective["points_for"].sum())
        points_against_total = float(perspective["points_against"].sum())
        point_share = points_for_total / max(points_for_total + points_against_total, 1.0)

        ue_proxy_series = self._estimate_unforced_error_proxy(
            attack_rate=perspective["attack_rate"],
            safe_rate=perspective["safe_rate"],
            flick_rate=perspective["flick_rate"],
            points_for=perspective["points_for"],
            points_against=perspective["points_against"],
        )
        unforced_error_rate = float((ue_proxy_series * rally_weight).sum() / rally_weight.sum())

        return_pressure = float(
            min(
                0.99,
                max(
                    0.01,
                    0.58 * base_rcv + 0.22 * attack + 0.20 * point_share,
                ),
            )
        )

        close_match = (perspective["points_for"] - perspective["points_against"]).abs() <= 6
        close_df = perspective.loc[close_match]
        if close_df.empty:
            clutch_point_win = point_share
        else:
            close_points_for = float(close_df["points_for"].sum())
            close_points_against = float(close_df["points_against"].sum())
            close_point_share = close_points_for / max(close_points_for + close_points_against, 1.0)
            close_win_rate = self._smooth_probability(float(close_df["won"].sum()), float(len(close_df)), alpha=1.0)
            clutch_point_win = 0.65 * close_point_share + 0.35 * close_win_rate
        clutch_point_win = float(min(0.99, max(0.01, clutch_point_win)))

        player_row = self.players_df[self.players_df["player_id"] == player_id].iloc[0]
        return {
            "player_id": player_id,
            "name": str(player_row["name"]),
            "matches": matches,
            "win_rate": self._smooth_probability(wins, matches, alpha=1.5),
            "base_srv_win": base_srv,
            "base_rcv_win": base_rcv,
            "unforced_error_rate": unforced_error_rate,
            "return_pressure": return_pressure,
            "clutch_point_win": clutch_point_win,
            "serve_mix": {"short": short, "flick": flick},
            "rally_style": {"attack": attack, "neutral": neutral, "safe": safe},
            "serve_trials": int(serve_trials),
            "receive_trials": int(receive_trials),
        }

    def get_head_to_head(
        self,
        player_a_id: str,
        player_b_id: str,
        window: int = 30,
        as_of_date: str | None = None,
    ) -> dict[str, Any]:
        df = self._window_filter(window=window * 2, as_of_date=as_of_date)
        h2h = df[
            ((df["playerA_id"] == player_a_id) & (df["playerB_id"] == player_b_id))
            | ((df["playerA_id"] == player_b_id) & (df["playerB_id"] == player_a_id))
        ].copy()

        if h2h.empty:
            return {
                "matches": 0,
                "a_win_rate": 0.5,
                "a_srv_win": 0.5,
                "a_rcv_win": 0.5,
            }

        perspective = self._perspective_frame(h2h, player_id=player_a_id)
        matches = int(len(perspective))
        wins = int(perspective["won"].sum())

        srv_trials = float(perspective["serve_rallies"].sum())
        srv_wins = float(perspective["serve_wins"].sum())
        rcv_trials = float(perspective["receive_rallies"].sum())
        rcv_wins = float(perspective["receive_wins"].sum())

        return {
            "matches": matches,
            "a_win_rate": self._smooth_probability(wins, matches, alpha=1.0),
            "a_srv_win": self._smooth_probability(srv_wins, srv_trials, alpha=1.5),
            "a_rcv_win": self._smooth_probability(rcv_wins, rcv_trials, alpha=1.5),
        }
