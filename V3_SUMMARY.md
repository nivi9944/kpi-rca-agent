# V3 summary

## Bottom line

- **v3 is a significant improvement over v1 and v2 on TEST**: top-1 47.9% vs 43.6% (v2) and 40.7% (v1); exact McNemar p = 0.0312 vs v2 (6 wins, 0 losses) and p = 0.002 vs v1 (10 wins, 0 losses).
- **It still ties the rule-based baseline B2**: p = 1.0 against B2 on the same (v3) engine, p = 0.2188 against B2 on the v2 engine. The gain comes mostly from the lower scan threshold, which lifts B2 by about as much.
- **DEV mostly failed**: top-1 55.0% to 52.5%, precision 41.1% to 38.3%, F1 47.9 to 46.0 (all down); recall, detection recall and price-drop top-1 unchanged.
- **The v3 changes (threshold, guard, price test) were chosen after reviewing v2's TEST results**, and v3 was evaluated on the same TEST set, so the v3 TEST numbers are **not a clean held-out estimate**.
- **That is why v2 remains the headline** in the README and the resume reference, and v3 is kept as a documented iteration, not a replacement.

## What changed in v3

1. Segment-scan threshold |hist_z| 4.25 to 4.0 (natural false-alarm rate 8.2% to 11.2%, results/calibration.json).
2. Candidate guard at submit_report: a cause must have a significant (BH) significance_test in the investigation's own trace; one correction message, then unverified causes are dropped.
3. Yuen's trimmed-mean test (20% trimmed) for money-metric rate tests (AOV, avg_item_price), aimed at price drops.
Engine switch in tools/engine.py keeps v2 reproducible. 1 worker throughout; v1, v2 and B1 results reused, B2 rerun.

## DEV verdict (50 DEV scenarios; PASS = at least as good as v2-final)

| Metric | v2-final | v3 | Verdict |
|---|---|---|---|
| Top-1 | 55.0% | 52.5% | FAIL |
| Precision | 41.1% | 38.3% | FAIL |
| Recall | 57.5% | 57.5% | PASS |
| F1 | 47.9% | 46.0% | FAIL |
| Detection recall | 65.0% | 65.0% | PASS |
| Price-drop top-1 | 14.3% | 14.3% | PASS |

## TEST results (140 planted + 30 controls; week-clustered bootstrap 95% CIs, Wilson in results/v3/summary.json)

| Investigator | Top-1 | Top-3 | MRR | Precision | Recall | F1 | Specificity | Detection | Top-1 once detected | Failed runs |
|---|---|---|---|---|---|---|---|---|---|---|
| Agent v3 (after TEST-informed changes) | 47.9% (67/140) [41.2, 55.5] | 51.4% | 0.4964 | 43.1% | 51.4% | 46.9 [43.6, 49.7] | 96.7% | 52.9% | 90.5% | 1 |
| B2: scan + test, v3 engine | 48.6% (68/140) [42.1, 55.7] | 49.3% | 0.4893 | 54.3% | 49.3% | 51.7 [48.1, 55.7] | 96.7% | 52.9% | 91.9% | 0 |
| Agent v2 (frozen) | 43.6% (61/140) [36.9, 50.7] | 46.4% | 0.4518 | 42.3% | 47.1% | 44.6 [40.9, 47.7] | 96.7% | 47.9% | 91.0% | 0 |
| Agent v1 (frozen) | 40.7% (57/140) [33.8, 48.9] | 44.3% | 0.4264 | 41.7% | 45.0% | 43.3 [38.9, 47.1] | 96.7% | 45.7% | 89.1% | 0 |
| B2: scan + test, v2 engine | 45.0% (63/140) [39.3, 51.5] | 45.7% | 0.4536 | 54.7% | 45.7% | 49.8 [46.0, 54.1] | 96.7% | 47.9% | 94.0% | 0 |
| B1: top contribution | 12.9% (18/140) [8.0, 17.5] | 15.7% | 0.1417 | 25.9% | 15.7% | 19.6 [14.5, 23.0] | 96.7% | 20.0% | 64.3% | 0 |

McNemar (exact, paired top-1): v3 vs v1: 10 vs 0, p = 0.002; v3 vs B2 (v3 engine): 0 vs 1, p = 1.0; v3 vs v2: 6 vs 0, p = 0.0312; v3 vs B2 (v2 engine): 5 vs 1, p = 0.2188; v2 vs v1: 4 vs 0, p = 0.125

Top-1 by type (v3 / B2 v3 engine / v2 / v1 / B2 v2 engine, out of 20):

- cancellation_spike: 15 / 15 / 14 / 14 / 14
- delay_then_reviews: 8 / 8 / 6 / 6 / 6
- delivery_delay: 7 / 7 / 6 / 6 / 6
- mix_shift: 19 / 20 / 18 / 15 / 20
- price_drop: 3 / 3 / 2 / 2 / 2
- review_drop: 8 / 8 / 8 / 7 / 8
- volume_drop: 7 / 7 / 7 / 7 / 7

Detection recall by severity (small / medium / large): v3 30.6% / 55.1% / 76.2%; v2 26.5% / 49.0% / 71.4%; B2 v3 engine 30.6% / 55.1% / 76.2%.

## Other v3 facts

- Grounding 99.59%; ungrounded numbers classified as {'derived': 10, 'invented': 1, 'threshold_notation': 1} (v2: {'derived': 5, 'threshold_notation': 3}, 0 invented).
- Candidate guard warned in 0 of 170 TEST runs: the agent always tested causes before submitting, so the guard had no effect on TEST.
- Price drops: 3/20 top-1 (v2: 2/20); the trimmed-mean test did not change the picture.
- Failed runs: 1 (t150_control_clean: timeout), a clean control week that hit the agent's 15-minute timeout after 2 steps (slow API); no report, so no false alarm.
- Cost per investigation: 217.54 s average (p95 321.77 s), 90033.4 tokens, 11.86 tool calls.

## Run notes

- Wall clock: started 2026-09-27 03:42; all scenario phases finished 16:31, inside the 15-hour ceiling (no time-budget stop).
- The gateway was down at the start (Docker not running after a sleep); the run was stopped before any v3 call, Docker and the gateway were brought back, and a pre-flight gateway check was added.
- The orchestrator crashed while writing the automatic draft summary (a key clash in eval/metrics_v3.py, now fixed); the metrics had already been computed correctly. This summary was written from the result files.

## Decision

v3 is reported as a documented iteration: detection went up and v3 beats v1 and v2 significantly on a reused TEST set, but it only ties B2 and DEV did not support it. v2 stays the headline result (README, release v2.0.0, resume). A clean claim for v3 would need a fresh TEST set built after these changes.
