## v3 DEV verdict

Same 50 DEV scenarios; rule: PASS if v3 is at least as good as v2-final. v3 DEV runs: 50 of 50 completed. Written automatically by eval/run_v3.py.

| Metric | v2-final (DEV) | v3 (DEV) | Delta | Verdict |
|---|---|---|---|---|
| Top-1 | 55.0% | 52.5% | -2.5 pts | FAIL |
| Precision | 41.1% | 38.3% | -2.8 pts | FAIL |
| Recall | 57.5% | 57.5% | +0.0 pts | PASS |
| F1 | 47.9% | 46.0% | -1.9 pts | FAIL |
| Detection recall | 65.0% | 65.0% | +0.0 pts | PASS |
| Price-drop top-1 | 14.3% | 14.3% | +0.0 pts | PASS |

For reference, B2 on DEV: top-1 52.5% (v2 engine) vs 52.5% (v3 engine); detection 65.0% vs 65.0%; specificity 100.0% vs 100.0%.
