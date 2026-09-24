# s45_control_clean (nvidia)

**Metric:** avg_review_score  
**Week:** 2017-07-31  
**Anomaly confirmed:** False  
**Change:** 0.05168%

## Root causes


## Narrative

The average review score in the week of 2017-07-31 was 4.217, virtually unchanged from the prior 4-week average of 4.215 (a +0.05% shift, z=0.25) [e1]. No segment showed a statistically significant and historically unusual change: the largest share shift was bed_bath_table dropping from 13.0% to 8.2% of orders (p=5.8e-5, but hist_z=-2.64 < 4.25) [e3], and the largest rate shift was customer_state PR falling from 4.49 to 3.99 (p=0.016, hist_z=-2.52) [e4]. All other candidates also failed the dual threshold (BH q≤0.05 and |hist_z|≥4.25) [e5–e12]. Therefore, no significant root cause is identified.

## Actions

- No action required for this metric this week; the headline is stable and no segment driver is significant.
- Continue routine monitoring; the detection threshold (|z|≥3) did not flag this week.

## Tool-call trace

- step 1: `detect_anomalies` {"metric":"avg_review_score","week":"2017-07-31"} -> e1
- step 2: `scan_segments` {"metric":"avg_review_score","week":"2017-07-31"} -> e2
- step 3: `significance_test` {"dimension":"product_category","segment":"bed_bath_table","metric":"avg_review_score","effect":"share","week":"2017-07-31"} -> e3
- step 4: `significance_test` {"effect":"rate","metric":"avg_review_score","dimension":"customer_state","week":"2017-07-31","segment":"PR"} -> e4
- step 5: `significance_test` {"dimension":"seller_state","effect":"share","week":"2017-07-31","segment":"MG","metric":"avg_review_score"} -> e5
- step 6: `significance_test` {"week":"2017-07-31","segment":"watches_gifts","effect":"share","metric":"avg_review_score","dimension":"product_category"} -> e6
- step 7: `significance_test` {"dimension":"product_category","segment":"auto","effect":"share","week":"2017-07-31","metric":"avg_review_score"} -> e7
- step 8: `significance_test` {"effect":"rate","metric":"avg_review_score","dimension":"seller_state","segment":"PR","week":"2017-07-31"} -> e8
- step 9: `significance_test` {"effect":"rate","segment":"SC","dimension":"customer_state","metric":"avg_review_score","week":"2017-07-31"} -> e9
- step 10: `significance_test` {"metric":"avg_review_score","dimension":"main_payment_type","segment":"voucher","effect":"rate","week":"2017-07-31"} -> e10
- step 11: `significance_test` {"dimension":"product_category","segment":"housewares","metric":"avg_review_score","effect":"rate","week":"2017-07-31"} -> e11
- step 12: `significance_test` {"dimension":"seller_state","segment":"SC","week":"2017-07-31","effect":"rate","metric":"avg_review_score"} -> e12

**Grounding:** 16/16 numbers verified (100.0%)