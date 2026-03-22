from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any


_CANONICAL_PARAM_KEYS = (
    "pA_srv_win",
    "pA_rcv_win",
    "serve_mix_A_short",
    "serve_mix_B_short",
    "rally_style_A_attack",
    "rally_style_B_attack",
    "rally_style_A_safe",
    "rally_style_B_safe",
    "unforced_error_A",
    "unforced_error_B",
    "return_pressure_A",
    "return_pressure_B",
    "clutch_A",
    "clutch_B",
)

_PARAM_KEY_LOOKUP = {key.lower(): key for key in _CANONICAL_PARAM_KEYS}
_LEGACY_PARAM_ALIASES = {
    "ue_rate_a": "unforced_error_A",
    "ue_rate_b": "unforced_error_B",
}
_PARAM_KEY_LOOKUP.update(_LEGACY_PARAM_ALIASES)

_PARAM_PATTERN = re.compile(
    r"\b(pA_srv_win|pA_rcv_win|serve_mix_A_short|serve_mix_B_short|"
    r"rally_style_A_attack|rally_style_B_attack|rally_style_A_safe|rally_style_B_safe|"
    r"unforced_error_A|unforced_error_B|ue_rate_A|ue_rate_B|"
    r"return_pressure_A|return_pressure_B|clutch_A|clutch_B)\b"
    r"\s*[:=]\s*([+-]?(?:\d*\.\d+|\d+))",
    flags=re.IGNORECASE,
)


def _logistic(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _normalize_param_key(key: str) -> str:
    return _PARAM_KEY_LOOKUP.get(key.lower(), key)


def _normalize_params(params: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for raw_key, value in params.items():
        key = _normalize_param_key(str(raw_key))
        if str(raw_key).lower() in _LEGACY_PARAM_ALIASES:
            normalized.setdefault(key, value)
            continue
        normalized[key] = value
    return normalized


def _get_param(params: dict[str, Any], key: str, *, legacy_key: str | None = None, default: float) -> float:
    if key in params:
        return float(params[key])
    if legacy_key is not None and legacy_key in params:
        return float(params[legacy_key])
    return default


def mock_probability(params: dict[str, Any]) -> float:
    """Deterministic monotonic mapping from rally params to match win probability."""

    params = _normalize_params(params)

    p_a_srv = _get_param(params, "pA_srv_win", default=0.5)
    p_a_rcv = _get_param(params, "pA_rcv_win", default=0.5)
    serve_edge = _get_param(params, "serve_mix_A_short", default=0.5) - _get_param(
        params, "serve_mix_B_short", default=0.5
    )
    attack_edge = _get_param(params, "rally_style_A_attack", default=0.33) - _get_param(
        params, "rally_style_B_attack", default=0.33
    )
    safe_edge = _get_param(params, "rally_style_B_safe", default=0.33) - _get_param(
        params, "rally_style_A_safe", default=0.33
    )
    ue_edge = _get_param(params, "unforced_error_B", legacy_key="ue_rate_B", default=0.18) - _get_param(
        params, "unforced_error_A", legacy_key="ue_rate_A", default=0.18
    )
    return_edge = _get_param(params, "return_pressure_A", default=0.5) - _get_param(
        params, "return_pressure_B", default=0.5
    )
    clutch_edge = _get_param(params, "clutch_A", default=0.5) - _get_param(params, "clutch_B", default=0.5)

    linear = (
        2.8 * (p_a_srv - 0.5)
        + 2.2 * (p_a_rcv - 0.5)
        + 0.7 * serve_edge
        + 0.9 * attack_edge
        - 0.6 * safe_edge
        + 0.9 * ue_edge
        + 0.7 * return_edge
        + 0.5 * clutch_edge
    )
    probability = _logistic(linear)
    return max(0.01, min(0.99, probability))


def _extract_params_from_pcsp(pcsp_text: str) -> dict[str, float]:
    params: dict[str, float] = {}
    for match in _PARAM_PATTERN.finditer(pcsp_text):
        raw_key = match.group(1)
        value = float(match.group(2))
        key = _normalize_param_key(raw_key)
        params[key] = value
    return params


def mock_run(pcsp_path: Path, out_path: Path) -> dict[str, Any]:
    pcsp_text = pcsp_path.read_text(encoding="utf-8", errors="replace")
    params = _extract_params_from_pcsp(pcsp_text)
    probability = mock_probability(params)

    out_text = (
        "PAT Mock Verification Result\n"
        f"with prob {probability:.6f}\n"
        "module: -pcsp\n"
    )
    out_path.write_text(out_text, encoding="utf-8")

    stdout = f"[mock] PAT finished for {pcsp_path.name}. Probability = {probability:.6f}\n"
    return {
        "ok": True,
        "returncode": 0,
        "cmd": ["mock_pat", "-pcsp", str(pcsp_path), str(out_path)],
        "stdout": stdout,
        "stderr": "",
        "probability": probability,
        "pat_out_path": str(out_path),
    }
