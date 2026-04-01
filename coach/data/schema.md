# Data Schema

## `sample_players.csv`
Columns:
- `player_id`: canonical stable ID
- `name`: display name
- `country`: country code/name
- `handedness`: `R` or `L`

## `sample_matches.csv`
Each row is one completed match from the perspective of the recorded `playerA_id` vs `playerB_id`.

Columns:
- `date`: match date (`YYYY-MM-DD`)
- `playerA_id`, `playerB_id`: participants
- `winner_id`: winner player ID
- `a_games_won`, `b_games_won`: games won in best-of-3
- `a_points`, `b_points`: total rally points won in match
- `a_serve_rallies`, `a_serve_wins`: A serve opportunities and won rallies
- `b_serve_rallies`, `b_serve_wins`: B serve opportunities and won rallies
- `a_short_serve_rate`, `a_flick_serve_rate`: A serve mix (sums to ~1)
- `a_attack_rate`, `a_neutral_rate`, `a_safe_rate`: A rally style mix (sums to ~1)
- `b_short_serve_rate`, `b_flick_serve_rate`: B serve mix (sums to ~1)
- `b_attack_rate`, `b_neutral_rate`, `b_safe_rate`: B rally style mix (sums to ~1)
- `tournament`: tournament name from ShuttleSet `match.csv`
- `round`: tournament round from ShuttleSet `match.csv`
- `duration_min`: match duration in minutes (`match.csv.duration`)
- `match_sets`: recorded number of sets (`match.csv.set`)
- `avg_rally_len`: average rally length, where each rally length is `max(ball_round)` in that rally; default `0.0` when no rallies
- `long_rally_share`: share of rallies with `rally_len >= 8`; default `0.0` when no rallies
- `a_backhand_rate`, `b_backhand_rate`: side-level backhand usage rate (`backhand == 1`); `NaN` treated as `0`
- `a_aroundhead_rate`, `b_aroundhead_rate`: side-level around-head usage rate (`aroundhead == 1`); `NaN` treated as `0`
- `a_net_error_lost_rate`, `b_net_error_lost_rate`: share of each side's lost rallies ending in net errors (`掛網`, `未過網`); default `0.0` if no lost rallies
- `a_out_error_lost_rate`, `b_out_error_lost_rate`: share of each side's lost rallies ending in out errors (`出界`); default `0.0` if no lost rallies
- `a_short_serve_win_rate`, `b_short_serve_win_rate`: rally win rate when serving short (`發短球` as first stroke); default `0.5` when no short serves
- `a_long_serve_win_rate`, `b_long_serve_win_rate`: rally win rate when serving long (`發長球` as first stroke); default `0.5` when no long serves
- `a_short_serve_samples`, `b_short_serve_samples`: number of short-serve rallies by side
- `a_long_serve_samples`, `b_long_serve_samples`: number of long-serve rallies by side

## Parameter Estimation Assumptions
- Base rally probabilities are estimated from historical serve/receive outcomes.
- Receive-win counts use opponent serve totals: `receive_wins = opp_serve_rallies - opp_serve_wins`.
- Laplace smoothing is applied to avoid 0/1 probabilities:
  - `p = (wins + alpha) / (trials + 2*alpha)`
- Strategy mixes are weighted averages with Dirichlet-style pseudocounts.
- Head-to-head is blended with player baseline (small-sample shrinkage).
- Additional fine-grained tactical proxies are derived from match-level stats:
  - `unforced_error_rate` (attack/flick/safe/point-loss proxy)
  - `return_pressure` (receive quality + attack intent + point share)
  - `clutch_point_win` (close-score point share + close-match win smoothing)
- New stable context/tactical features are additionally available for PAT parameterization:
  - serve-type effectiveness (`short_serve_win_rate`, `long_serve_win_rate` + sample counts)
  - rally tolerance (`avg_rally_len`, `long_rally_share`)
  - stroke profile (`backhand_rate`, `aroundhead_rate`)
  - terminal error profile (`net_error_lost_rate`, `out_error_lost_rate`)
  - handedness and reliability scaling from `sample_players.csv` plus serve/receive trial volume
- Weight calibration uses pre-match player-history snapshots to reduce same-match leakage when fitting influence weights.

## Limits
- Match-level proxies are used instead of full rally logs.
- Style effects (`w_short`, `w_attack`, `w_safe`) are estimated from historical aggregate trends.
- Some tactical features are derived from sparse terminal-event labels (`lose_reason`, `win_reason`) and should be interpreted as directional signals, not exhaustive causal attribution.
