from __future__ import annotations

import csv
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from coach.config import CoachConfig
from coach.data.adapters.local_csv import LocalCSVAdapter
from coach.data.stats_builder import MatchupStats, build_matchup_params
from coach.model.builder import ModelBuildResult, build_matchup_model
from coach.model.params import MatchupParams
from coach.pat.parser import parse_probability, read_pat_output
from coach.pat.runner import run_pat
from coach.runs import new_run_dir
from coach.utils import write_json


@dataclass(frozen=True)
class PATExecution:
    ok: bool
    returncode: int
    cmd: list[str]
    stdout_path: Path
    stderr_path: Path
    pat_out_path: Path
    probability: float | None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ok": self.ok,
            "returncode": self.returncode,
            "cmd": self.cmd,
            "stdout_path": str(self.stdout_path),
            "stderr_path": str(self.stderr_path),
            "pat_out_path": str(self.pat_out_path),
            "probability": self.probability,
        }
        if self.error is not None:
            payload["error"] = self.error
        return payload


@dataclass(frozen=True)
class PredictionResult:
    run_id: str
    run_dir: Path
    mode: str
    player_a: str
    player_b: str
    probability: float
    params: MatchupParams
    stats: MatchupStats
    model: ModelBuildResult
    pat: PATExecution


@dataclass(frozen=True)
class StrategyCandidate:
    rank: int
    serve_short_delta: float
    attack_delta: float
    unforced_error_delta: float
    return_pressure_delta: float
    clutch_delta: float
    l1_change: float
    probability: float


@dataclass(frozen=True)
class StrategyResult:
    run_id: str
    run_dir: Path
    mode: str
    player_a: str
    player_b: str
    baseline_probability: float
    improved_probability: float
    delta: float
    best_candidate: StrategyCandidate
    top_alternatives: list[StrategyCandidate]
    params_baseline: MatchupParams
    params_best: MatchupParams


class BadmintonCoachService:
    """End-to-end execution service used by CLI, analysis scripts, and agent tools."""

    def __init__(
        self,
        adapter: LocalCSVAdapter | None = None,
        template_name: str = "badminton_rally_template.pcsp",
        runs_root: str | Path | None = None,
        config: CoachConfig | None = None,
    ) -> None:
        self.config = config or CoachConfig.from_env()
        self.adapter = adapter or LocalCSVAdapter()
        self.template_name = template_name
        self.runs_root = Path(runs_root) if runs_root is not None else self.config.runs_dir

    def predict(
        self,
        player_a: str,
        player_b: str,
        window: int = 30,
        mode: str = "mock",
        run_id: str | None = None,
        pat_path: str | None = None,
        timeout_s: int | None = None,
        as_of_date: str | None = None,
    ) -> PredictionResult:
        run_name, run_dir = self._make_run_dir(prefix="predict", run_id=run_id)

        params, stats = build_matchup_params(
            adapter=self.adapter,
            player_a_ref=player_a,
            player_b_ref=player_b,
            window=window,
            as_of_date=as_of_date,
        )

        model = build_matchup_model(
            params=params,
            template_name=self.template_name,
            out_path=run_dir / "matchup.pcsp",
        )

        pat_exec = self._execute_pat(
            pcsp_path=model.matchup_pcsp_path,
            run_dir=run_dir,
            mode=mode,
            pat_path=pat_path,
            timeout_s=timeout_s,
        )
        if pat_exec.probability is None:
            raise RuntimeError(
                "PAT run completed but probability parsing failed. "
                f"See {pat_exec.pat_out_path} and {pat_exec.stdout_path}."
            )

        result = PredictionResult(
            run_id=run_name,
            run_dir=run_dir,
            mode=mode,
            player_a=params.player_a.name,
            player_b=params.player_b.name,
            probability=pat_exec.probability,
            params=params,
            stats=stats,
            model=model,
            pat=pat_exec,
        )

        self._write_prediction_artifacts(result=result, window=window, as_of_date=as_of_date)
        return result

    def strategy(
        self,
        player_a: str,
        player_b: str,
        window: int = 30,
        mode: str = "mock",
        budget: int = 60,
        l1_bound: float = 0.3,
        run_id: str | None = None,
        pat_path: str | None = None,
        timeout_s: int | None = None,
        as_of_date: str | None = None,
    ) -> StrategyResult:
        run_name, run_dir = self._make_run_dir(prefix="strategy", run_id=run_id)

        params, stats = build_matchup_params(
            adapter=self.adapter,
            player_a_ref=player_a,
            player_b_ref=player_b,
            window=window,
            as_of_date=as_of_date,
        )

        baseline_model = build_matchup_model(
            params=params,
            template_name=self.template_name,
            out_path=run_dir / "baseline.pcsp",
        )

        baseline_pat = self._execute_pat(
            pcsp_path=baseline_model.matchup_pcsp_path,
            run_dir=run_dir,
            mode=mode,
            pat_path=pat_path,
            timeout_s=timeout_s,
        )
        if baseline_pat.probability is None:
            raise RuntimeError(
                "Could not parse baseline probability from PAT output. "
                f"See {baseline_pat.pat_out_path}."
            )

        candidates = self._generate_candidates(params, l1_bound=l1_bound)
        candidates = candidates[: max(1, budget)]

        ranked: list[tuple[StrategyCandidate, MatchupParams]] = []
        candidate_dir = run_dir / "candidates"
        candidate_dir.mkdir(parents=True, exist_ok=True)

        for idx, candidate in enumerate(candidates, start=1):
            adjusted = params.with_adjustments(
                serve_short_delta=candidate["serve_short_delta"],
                attack_delta=candidate["attack_delta"],
                unforced_error_delta=candidate["unforced_error_delta"],
                return_pressure_delta=candidate["return_pressure_delta"],
                clutch_delta=candidate["clutch_delta"],
            )

            cand_path = candidate_dir / f"candidate_{idx:03d}.pcsp"
            build = build_matchup_model(
                params=adjusted,
                template_name=self.template_name,
                out_path=cand_path,
            )

            cand_run_dir = candidate_dir / f"candidate_{idx:03d}"
            pat_exec = self._execute_pat(
                pcsp_path=build.matchup_pcsp_path,
                run_dir=cand_run_dir,
                mode=mode,
                pat_path=pat_path,
                timeout_s=timeout_s,
            )
            if pat_exec.probability is None:
                continue

            ranked.append(
                (
                    StrategyCandidate(
                        rank=0,
                        serve_short_delta=adjusted.player_a.serve_mix.short - params.player_a.serve_mix.short,
                        attack_delta=adjusted.player_a.rally_style.attack - params.player_a.rally_style.attack,
                        unforced_error_delta=adjusted.player_a.unforced_error_rate
                        - params.player_a.unforced_error_rate,
                        return_pressure_delta=adjusted.player_a.return_pressure - params.player_a.return_pressure,
                        clutch_delta=adjusted.player_a.clutch_point_win - params.player_a.clutch_point_win,
                        l1_change=adjusted.l1_change_from(params),
                        probability=pat_exec.probability,
                    ),
                    adjusted,
                )
            )

        if not ranked:
            raise RuntimeError("No strategy candidates yielded parseable probabilities.")

        ranked.sort(key=lambda x: x[0].probability, reverse=True)
        top = ranked[:5]

        best_candidate, best_params = ranked[0]
        improved_probability = best_candidate.probability
        baseline_probability = baseline_pat.probability

        numbered = [
            StrategyCandidate(
                rank=i,
                serve_short_delta=c.serve_short_delta,
                attack_delta=c.attack_delta,
                unforced_error_delta=c.unforced_error_delta,
                return_pressure_delta=c.return_pressure_delta,
                clutch_delta=c.clutch_delta,
                l1_change=c.l1_change,
                probability=c.probability,
            )
            for i, (c, _) in enumerate(top, start=1)
        ]

        result = StrategyResult(
            run_id=run_name,
            run_dir=run_dir,
            mode=mode,
            player_a=params.player_a.name,
            player_b=params.player_b.name,
            baseline_probability=baseline_probability,
            improved_probability=improved_probability,
            delta=improved_probability - baseline_probability,
            best_candidate=numbered[0],
            top_alternatives=numbered,
            params_baseline=params,
            params_best=best_params,
        )

        self._write_strategy_artifacts(result=result, stats=stats, window=window, as_of_date=as_of_date)
        return result

    def _execute_pat(
        self,
        *,
        pcsp_path: Path,
        run_dir: Path,
        mode: str,
        pat_path: str | None,
        timeout_s: int | None,
    ) -> PATExecution:
        out_path = run_dir / "pat_output.txt"
        pat_console = Path(pat_path).expanduser() if pat_path else self.config.pat_console_path
        resolved_timeout = timeout_s or self.config.pat_timeout_s
        use_mono = self.config.resolve_use_mono(pat_console)

        result = run_pat(
            pcsp_path=pcsp_path,
            out_path=out_path,
            mode=mode,
            pat_console_path=pat_console,
            timeout_s=resolved_timeout,
            use_mono=use_mono,
        )

        probability: float | None = None
        parse_error: str | None = None
        try:
            probability = parse_probability(read_pat_output(out_path))
        except Exception as exc:
            parse_error = str(exc)

        error = None
        if not result.get("ok", False):
            error = str(result.get("error", "PAT execution failed"))
        elif parse_error is not None:
            error = parse_error

        return PATExecution(
            ok=bool(result.get("ok", False)),
            returncode=int(result.get("returncode", -1)),
            cmd=[str(part) for part in result.get("cmd", [])],
            stdout_path=Path(str(result.get("stdout_path"))),
            stderr_path=Path(str(result.get("stderr_path"))),
            pat_out_path=Path(str(result.get("pat_out_path"))),
            probability=probability,
            error=error,
        )

    def _make_run_dir(self, prefix: str, run_id: str | None) -> tuple[str, Path]:
        if run_id is not None:
            run_dir = self.runs_root / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            return run_id, run_dir
        return new_run_dir(prefix=prefix, base_dir=self.runs_root)

    def _generate_candidates(self, baseline: MatchupParams, l1_bound: float) -> list[dict[str, float]]:
        knob_steps: dict[str, list[float]] = {
            "serve_short_delta": [-0.08, -0.05, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.05, 0.08],
            "attack_delta": [-0.08, -0.05, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.05, 0.08],
            "unforced_error_delta": [-0.06, -0.04, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.04, 0.06],
            "return_pressure_delta": [-0.06, -0.04, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.04, 0.06],
            "clutch_delta": [-0.04, -0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.04],
        }
        pair_knobs = [
            ("serve_short_delta", "attack_delta"),
            ("serve_short_delta", "unforced_error_delta"),
            ("serve_short_delta", "return_pressure_delta"),
            ("attack_delta", "unforced_error_delta"),
            ("attack_delta", "clutch_delta"),
            ("return_pressure_delta", "clutch_delta"),
        ]
        pair_steps = {k: [d for d in values if abs(d) <= 0.03] for k, values in knob_steps.items()}

        candidates: list[dict[str, float]] = []
        seen: set[tuple[float, ...]] = set()
        ordered_knobs = list(knob_steps.keys())

        def _make_payload(changes: dict[str, float]) -> dict[str, float]:
            payload = {k: 0.0 for k in ordered_knobs}
            payload.update(changes)
            return payload

        def _maybe_add(changes: dict[str, float]) -> None:
            payload = _make_payload(changes)
            key = tuple(round(payload[k], 6) for k in ordered_knobs)
            if key in seen:
                return

            adjusted = baseline.with_adjustments(
                serve_short_delta=payload["serve_short_delta"],
                attack_delta=payload["attack_delta"],
                unforced_error_delta=payload["unforced_error_delta"],
                return_pressure_delta=payload["return_pressure_delta"],
                clutch_delta=payload["clutch_delta"],
            )
            l1 = adjusted.l1_change_from(baseline)
            if l1 <= l1_bound + 1e-9:
                payload["l1_change"] = l1
                candidates.append(payload)
                seen.add(key)

        for knob, steps in knob_steps.items():
            for delta in steps:
                _maybe_add({knob: delta})

        for knob_1, knob_2 in pair_knobs:
            for delta_1 in pair_steps[knob_1]:
                for delta_2 in pair_steps[knob_2]:
                    _maybe_add({knob_1: delta_1, knob_2: delta_2})

        def _magnitude(candidate: dict[str, float]) -> float:
            return sum(abs(candidate[k]) for k in ordered_knobs)

        candidates.sort(key=lambda c: (c["l1_change"], _magnitude(c)))
        return candidates

    def _write_prediction_artifacts(self, result: PredictionResult, window: int, as_of_date: str | None) -> None:
        write_json(
            result.run_dir / "inputs.json",
            {
                "task": "prediction",
                "player_a": result.player_a,
                "player_b": result.player_b,
                "mode": result.mode,
                "window": window,
                "as_of_date": as_of_date,
                "run_id": result.run_id,
            },
        )
        write_json(
            result.run_dir / "stats.json",
            {
                "player_a": result.stats.player_a_stats,
                "player_b": result.stats.player_b_stats,
                "head_to_head": result.stats.head_to_head,
                "weights": result.stats.weights.model_dump(),
            },
        )
        write_json(
            result.run_dir / "prediction_result.json",
            {
                "run_id": result.run_id,
                "probability": result.probability,
                "player_a": result.player_a,
                "player_b": result.player_b,
                "mode": result.mode,
                "pcsp_path": str(result.model.matchup_pcsp_path),
                "pat": result.pat.to_dict(),
            },
        )
        write_json(
            result.run_dir / "summary.json",
            {
                "question": f"Predict win probability: {result.player_a} vs {result.player_b}",
                "players": [result.player_a, result.player_b],
                "params_used": result.params.model_dump(),
                "probability": result.probability,
                "timestamps": {
                    "generated_utc": dt.datetime.now(dt.UTC).isoformat(),
                },
            },
        )

    def _write_strategy_artifacts(
        self,
        result: StrategyResult,
        stats: MatchupStats,
        window: int,
        as_of_date: str | None,
    ) -> None:
        write_json(
            result.run_dir / "inputs.json",
            {
                "task": "strategy",
                "player_a": result.player_a,
                "player_b": result.player_b,
                "mode": result.mode,
                "window": window,
                "as_of_date": as_of_date,
                "run_id": result.run_id,
            },
        )
        write_json(
            result.run_dir / "stats.json",
            {
                "player_a": stats.player_a_stats,
                "player_b": stats.player_b_stats,
                "head_to_head": stats.head_to_head,
                "weights": stats.weights.model_dump(),
            },
        )
        write_json(
            result.run_dir / "strategy_result.json",
            {
                "run_id": result.run_id,
                "baseline_probability": result.baseline_probability,
                "improved_probability": result.improved_probability,
                "delta": result.delta,
                "best_candidate": result.best_candidate.__dict__,
                "top_alternatives": [c.__dict__ for c in result.top_alternatives],
            },
        )
        write_json(
            result.run_dir / "summary.json",
            {
                "question": f"Strategy optimization: {result.player_a} vs {result.player_b}",
                "players": [result.player_a, result.player_b],
                "params_used": result.params_baseline.model_dump(),
                "probability": {
                    "baseline": result.baseline_probability,
                    "improved": result.improved_probability,
                    "delta": result.delta,
                },
                "timestamps": {
                    "generated_utc": dt.datetime.now(dt.UTC).isoformat(),
                },
            },
        )

        csv_path = result.run_dir / "top_alternatives.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "rank",
                    "serve_short_delta",
                    "attack_delta",
                    "unforced_error_delta",
                    "return_pressure_delta",
                    "clutch_delta",
                    "l1_change",
                    "probability",
                ],
            )
            writer.writeheader()
            for cand in result.top_alternatives:
                writer.writerow(cand.__dict__)
