# s22_delivery_delay_small (nvidia)

**Metric:** on_time_rate  
**Week:** 2017-07-17  
**Anomaly confirmed:** True  
**Change:** -3.04%

## Root causes

1. seller_state = SP | effect rate | contribution 112.59% | p 5.255e-08 | impact R$ None | evidence e3, e5, e7
2. product_category = garden_tools | effect mix | contribution -6.51% | p 4.627e-11 | impact R$ None | evidence e4, e6

## Narrative

The on-time delivery rate dropped 3.0 percentage points to 94.1% in the week of 2017-07-17 versus the prior 4-week average of 97.1% (e1). The dominant driver was a sharp decline in São Paulo sellers' on-time rate from 97.3% to 92.9%, accounting for 112.6% of the total change (e3, e7). A secondary factor was a mix shift: garden_tools orders rose from 2.7% to 7.4% of volume, partially offsetting the decline by -6.5% (e4, e6). No other segment reached statistical significance.

## Actions

- Investigate São Paulo seller logistics (carrier capacity, warehouse staffing) for the week of 2017-07-17 to identify the root cause of the 4.4 pp on-time drop.
- Monitor garden_tools order share; if the mix shift persists, ensure its typically lower on-time rate does not drag the overall metric further.

## Tool-call trace

- step 1: `detect_anomalies` {"metric":"on_time_rate","week":"2017-07-17"} -> e1
- step 2: `scan_segments` {"metric":"on_time_rate","week":"2017-07-17"} -> e2
- step 3: `significance_test` {"dimension":"seller_state","effect":"rate","metric":"on_time_rate","segment":"SP","week":"2017-07-17"} -> e3
- step 4: `significance_test` {"dimension":"product_category","week":"2017-07-17","effect":"share","segment":"garden_tools","metric":"on_time_rate"} -> e4
- step 5: `estimate_impact` {"week":"2017-07-17","dimension":"seller_state","metric":"on_time_rate","segment":"SP"} -> e5
- step 6: `estimate_impact` {"dimension":"product_category","segment":"garden_tools","week":"2017-07-17","metric":"on_time_rate"} -> e6
- step 7: `drill_down` {"week":"2017-07-17","dimension":"seller_state","top_k":5,"metric":"on_time_rate"} -> e7
- step 8: `drill_down` {"metric":"on_time_rate","dimension":"product_category","top_k":5,"week":"2017-07-17"} -> e8
- step 9: `significance_test` {"segment":"computers_accessories","dimension":"product_category","effect":"rate","metric":"on_time_rate","week":"2017-07-17"} -> e9
- step 10: `significance_test` {"effect":"rate","segment":"bed_bath_table","dimension":"product_category","metric":"on_time_rate","week":"2017-07-17"} -> e10
- step 11: `significance_test` {"metric":"on_time_rate","week":"2017-07-17","effect":"rate","segment":"health_beauty","dimension":"product_category"} -> e11
- step 12: `significance_test` {"dimension":"product_category","metric":"on_time_rate","effect":"rate","week":"2017-07-17","segment":"sports_leisure"} -> e12

**Grounding:** 14/14 numbers verified (100.0%)