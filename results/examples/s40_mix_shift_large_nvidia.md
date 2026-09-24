# s40_mix_shift_large (nvidia)

**Metric:** aov  
**Week:** 2017-10-30  
**Anomaly confirmed:** True  
**Change:** -10.7644%

## Root causes

1. product_category = fashion_bags_accessories | effect mix | contribution 144.79% | p 5.309e-179 | impact R$ None | evidence e3, e4
2. seller_state = MG | effect mix | contribution 44.9% | p 6.455e-19 | impact R$ None | evidence e3, e5
3. seller_state = MG | effect rate | contribution 44.9% | p 0.002436 | impact R$ None | evidence e3, e8
4. is_repeat_customer = 0 | effect mix | contribution 66.46% | p 4.002e-14 | impact R$ None | evidence e3, e6
5. is_repeat_customer = 1 | effect mix | contribution 33.54% | p 4.002e-14 | impact R$ None | evidence e3, e7

## Narrative

AOV dropped 10.8% to R$150.04 in the week of 2017-10-30 versus the prior 4-week average of R$168.14 (e1). The decline was driven entirely by a 11.1% fall in value per item (e2). Eight segment-level shifts were statistically significant (e3–e11). The largest mix effect came from fashion_bags_accessories, whose order share surged from 2.2% to 27.2% (e4). Seller state MG doubled its share from 8.5% to 17.4% and its AOV fell from R$156 to R$115 (e5, e8). New customers (is_repeat_customer=0) rose from 96.6% to 91.5% of orders (e6), while repeat customers dropped from 3.4% to 8.5% share and their AOV fell from R$134 to R$83 (e7, e10). Voucher payment share nearly doubled (e9) and SP seller share fell (e11). Drill-down shows the aggregate rate effect was -R$26.13, dominated by small segments (e12).

## Actions

- Investigate the fashion_bags_accessories surge: check for a promotion, new seller onboarding, or catalog change that shifted mix toward lower-priced items.
- Review MG seller acquisition and pricing: the share jump and AOV drop suggest new sellers or discounting in that state.
- Assess new-customer acquisition quality: the sharp rise in new-customer share and their lower AOV may indicate a marketing channel bringing lower-value buyers.
- Monitor voucher usage: the share increase could reflect a coupon campaign that depresses basket value.

## Tool-call trace

- step 1: `detect_anomalies` {"week":"2017-10-30","metric":"aov"} -> e1
- step 2: `decompose_metric` {"metric":"aov","week":"2017-10-30"} -> e2
- step 3: `scan_segments` {"week":"2017-10-30","metric":"aov"} -> e3
- step 4: `significance_test` {"week":"2017-10-30","effect":"share","segment":"fashion_bags_accessories","metric":"aov","dimension":"product_category"} -> e4
- step 5: `significance_test` {"week":"2017-10-30","segment":"MG","metric":"aov","effect":"share","dimension":"seller_state"} -> e5
- step 6: `significance_test` {"effect":"share","metric":"aov","segment":"0","dimension":"is_repeat_customer","week":"2017-10-30"} -> e6
- step 7: `significance_test` {"metric":"aov","segment":"1","dimension":"is_repeat_customer","week":"2017-10-30","effect":"share"} -> e7
- step 8: `significance_test` {"effect":"rate","dimension":"seller_state","segment":"MG","week":"2017-10-30","metric":"aov"} -> e8
- step 9: `significance_test` {"segment":"voucher","metric":"aov","week":"2017-10-30","effect":"share","dimension":"main_payment_type"} -> e9
- step 10: `significance_test` {"segment":"1","week":"2017-10-30","dimension":"is_repeat_customer","metric":"aov","effect":"rate"} -> e10
- step 11: `significance_test` {"segment":"SP","dimension":"seller_state","week":"2017-10-30","metric":"aov","effect":"share"} -> e11
- step 12: `drill_down` {"metric":"aov","dimension":"product_category","week":"2017-10-30","top_k":10} -> e12

**Grounding:** 28/28 numbers verified (100.0%)