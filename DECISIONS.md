# Design decisions

The main design choices and the reason for each. Headline results are in the [README](README.md).

## Data

| Decision | Why |
|---|---|
| Olist public data, window 2017-01-01 to 2018-08-31, weekly grain (weeks start Monday) | Real multi-table e-commerce data; both ends of the full range are sparse |
| Two fact tables: `fact_orders` (one row per order) and `fact_items` (one row per item) | GMV and category/seller drill-downs need items; rates need orders. Items also carry the order-level dimensions, so GMV can be sliced by customer state or payment type |
| Order-level metrics use the order's **main item** (largest value) for category, seller and seller state | An order can span categories; assigning it once keeps segment shares summing to 100% |
| Main payment type = type with the largest paid value | Orders can have several payment rows; this avoids double counting |
| Repeat-customer flag computed on the full history before cutting the window | Otherwise early-2017 repeat customers would look new |
| Orders without items dropped (about 0.8%) | They carry no GMV and no category |
| Canceled/unavailable orders excluded from GMV, Orders, AOV; counted in cancellation rate | Standard GMV definition |

## Metrics and tools

| Decision | Why |
|---|---|
| Every metric is `sum` or `mean` over filtered rows (`metrics/catalog.yaml`) | One engine computes all 8 metrics and all drill-downs; each metric is defined once |
| Baseline = previous 4 weeks (average weekly sum, or pooled mean) | Smooths week-to-week noise; a single prior week is too noisy |
| Headline detector: robust z of the change vs the prior-4 average, against the 8 trailing weekly changes, median/MAD, flag at abs(z) >= 3 | Median/MAD is not distorted by past spikes (Black Friday); scoring the change removes Olist's strong growth trend |
| GMV = Orders x AOV with a **log split** | Exact and order-independent: ln(G1/G0) = ln(O1/O0) + ln(A1/A0) |
| Mix vs rate split with the mix term **centered on the overall baseline rate** | Sums to the same total, but each segment's mix term becomes meaningful: gaining share only lowers the metric if the segment is below average. This is how Simpson-style mix shifts are separated from real behaviour changes |
| **Segment scan** added to the spec's tool list | Found during evaluation: a problem in one segment is often invisible in the total (a 50% order drop in RJ moves total GMV about 6%, inside normal noise). Each segment is scored against its own history instead |
| Segment scan uses row shares (not value shares) and log(1 + value) for money rates | Value shares and AOV swing with a few big-ticket orders; found by inspecting baseline misses |
| Scan noise floor = binomial / standard-error noise expected from the segment's size, with Laplace smoothing | A 40-order segment cannot look unusual from a handful of orders; a baseline rate of exactly 0 (cancellations) still gets a sensible scale |
| Significance: chi-square on share (GMV/Orders, mix), two-proportion z (binary rates), Welch t (continuous), plus Mann-Whitney reported | Share tests are robust to overall growth: they ask whether the segment moved differently from everything else |
| A root cause must pass **BH q <= 0.05 AND abs(hist_z) >= 4.25** | Classical tests treat every order as independent; real weekly data is over-dispersed, so on real weeks they call nearly every large segment significant. The history check fixes that; BH controls false discoveries across the many segments tested |
| Threshold 4.25 chosen on **natural weeks only** (`eval/calibrate.py`): smallest threshold with at most 10% of natural week x metric pairs raising an alarm | Not tuned on the planted scenarios. Honest caveat: the scan design (row shares, log values, noise floor) was iterated while looking at baseline misses; the threshold itself never saw them |
| SQL tool: single read-only SELECT/WITH, keyword blocklist (incl. file readers), 1,000-row cap, 10 s timeout, runs on an in-memory copy | Lets the agent check things ad hoc without any way to write or read files |

## Agent

| Decision | Why |
|---|---|
| Own loop, plain OpenAI-style function calling, no LangChain | Every step is explainable; fewer dependencies |
| Final answer through a `submit_report` tool whose parameters are the Pydantic `Report` schema | Structured output that is validated; invalid reports go back to the model with the error |
| Grounding verifier: every number in the narrative and root causes must match a **cited** tool output within the precision written | Enforces "numbers only from tools"; one corrective retry, then the report is flagged ungrounded |
| Tool schemas flattened (no free-form objects, no min/max keywords) | Some OpenAI-compatible endpoints (Gemini's first) accept only a subset of JSON Schema; Pydantic still validates |
| Temperature 0, max 12 tool calls, token budget, timeout | Reproducible, bounded cost; the gateway only caches temperature-0 calls |
| Investigation id in the system prompt | The gateway's semantic cache matches on the last user message among requests with the same earlier messages. Two scenarios that differ only by week could otherwise share a cached first step. Exact-cache reruns still hit |
| Exact model name pinned per preset (`nvidia/nemotron-3-ultra-550b-a55b`, `qwen2.5:7b`, ...) and recorded in every run row, with the name the provider reports back | The gateway cache key includes the model name, so runs of different models never share cached answers; each row proves which model answered |
| Every preset forces its provider (`X-Provider-Force`), no fallback | With fallback enabled, a rate-limited call can be answered by a different model (for example local Qwen), so one run could silently mix models. Forcing makes each run single-model |
| Tool results sent to the model as compact JSON; exact echoes of the call's own arguments and repeated long guidance text (25+ characters, first copy kept) dropped (`agent/compact.py`) | Fewer tokens (measured about 6% on a stored run) with no information lost. Only the model's view is compacted: `inv.evidence` keeps full outputs and the verifier checks against those, so grounding cannot change |

## Evaluation

| Decision | Why |
|---|---|
| 40 planted scenarios (6 types x 3 severities) + 10 controls (5 clean, 5 with 5% uniform random order loss), fixed seeds | Ground truth is known; controls measure false alarms |
| Severities calibrated **per type** (e.g. delays of 5/10/15 days, volume drops of 30/50/80%) instead of one 10/20/35% scale | Olist delivers about 11 days before its estimate, so a 2-day delay changes almost nothing; one shared scale would make whole types invisible |
| Target weeks exclude Black Friday/holidays, the early-2018 delivery crisis, the May 2018 truckers' strike, the World Cup dip, and weeks where the headline is already unusual (abs(z) >= 2) | Real events would pollute the ground truth |
| Control weeks are NOT filtered by the detector | Conservative: a real event in a control week counts against us |
| Top-1 requires dimension + segment (+ effect type for rate/mix cases) | Getting the segment right but calling a mix shift a rate change is wrong |
| Two baselines: B1 (spec rule: top contribution if headline anomalous) and B2 (scan + tests, no LLM) | B1 is the naive analyst; B2 isolates what the LLM adds on top of the same tools |
| Resumable JSONL runs, throttle, backoff on 429 | Free-tier limits |
| A scenario whose LLM calls failed after all retries (API down, quota) is retried on the next run; model failures such as "no report after the step limit" are scored as failures, not retried; failed attempts stay in the file and are counted as `infra_failed_attempts`; only the latest attempt per scenario is scored | An outage is not a model failure, but hiding it would be dishonest. The run stops after 2 such scenarios in a row (likely a daily quota) |
| Headline cost reported as **$0 actual plus tokens per run** | The NVIDIA API catalog free tier has no list price. The Gemini and Mistral presets keep an estimated cost at their paid list prices (Gemini Flash $0.75 / $3.75, Mistral Small 4 $0.15 / $0.60, Mistral Medium 3.5 $1.50 / $7.50 per 1M tokens) |
| Spending guard: `BUDGET_USD` (8) on the estimated list-price spend of every run file in `results/` | Protects any paid comparison run; it never triggers for the free NVIDIA runs |
| The gateway's own `est_cost_usd` log field uses the Gemini list price for every provider | Left as it is: the gateway is a separate project, and all costs reported here come from the agent's own per-model accounting |
| A model run on fewer scenarios than the manifest is labelled "subset" and kept out of the charts and the per-type table | Two Gemini smoke-test rows stay in `results/` for transparency, but must not read as a full result |

## v2 changes

Each change is general (not tied to any scenario) and was developed on the 50-scenario DEV set only; the
held-out TEST set was built and run once after the code was frozen.

| Decision | Why |
|---|---|
| **Drop guard** (`agent/guards.py`): after the report passes the verifier, a root cause with no significant test on the same dimension and segment is removed; the LLM's order is kept | A cause the tools never confirmed should not reach the report. Which test counts: cited tests first, then tests whose effect matches the label (mix = share, rate = rate) |
| **Consistency guard**: if a significant cause remains, `anomaly_confirmed` is set to true; if none remains, `root_causes` is empty | A correctness fix: on DEV the model once named the right, significant cause but set anomaly_confirmed=false (s06) |
| **Evidence re-ranking tried and rejected** (`agent/rerank.py`, `--reorder`, off by default): order causes by significance, then \|hist_z\|, then q | On DEV it changed 2 of 50 reports and both got worse: s22 (garden_tools mix moved above the true seller_state SP rate) and s24 (customer_state SP moved above the true main_payment_type credit_card). \|hist_z\| measures how unusual a segment is, not how much of the KPI change it explains, so this rule inherits the statistical baseline's mistakes. DEV top-1: 50.0% with re-ordering vs 55.0% without, on the same LLM outputs |
| The LLM's raw report is stored next to the guarded report in every run row (`report_raw`), with the full evidence | Any later ranking or guard idea can be scored on stored runs without new API calls |
| **Prompt rules 7 and 8**: rank causes by evidence strength and label mix/rate by the matching test; for AOV, also scan avg_item_price. A longer price-drop rule (decompose first, then drill avg_item_price by category) was tried in DEV round 2 and **rejected**: price-drop top-1 stayed at 1 of 7 and the agent used avg_item_price in only 2 of 50 runs | Makes the LLM's own ordering and labels consistent with the tools; kept to two short sentences |
| **v1 reproduction switch** (`--agent-version v1`): the v1 prompt verbatim, `avg_item_price` hidden from the tools, no re-ranking | Lets v1 and v2 run on the same new TEST scenarios with the same injection code, so the comparison is fair |
| Faster pacing: gateway per-key limit raised to about 24 requests/min (`REFILL_PER_SEC=0.4`, `BUCKET_CAPACITY=20`); the nvidia client paces at 6/min | The old 10/min gateway limit made a full run take about two hours. 20/min and then 12/min were tried, but NVIDIA returned frequent 429s (enough to trip the gateway circuit breaker; only about 5 calls/min succeeded), so the pace is 6/min |
| **Step limit 15 for v2** (v1 keeps 12) | 34 of 50 v2 DEV round-1 runs (68%) hit the limit of 12 (v1: 28 of 50, 56%); the rule was to raise it if more than 10% did |
| **"v2 + B2 fallback"** scored offline as a separate investigator, never merged into v2: when v2 names no significant cause and B2 found one, B2's top cause is used (flagged `source=fallback`) | Shows what a simple hybrid would add without changing the agent; reported next to v2 on TEST |

## Held-out TEST set

| Decision | Why |
|---|---|
| DEV = the original 50 scenarios (used freely for debugging); TEST = `inject/manifest_test.json`, new master seed, built after v2 was frozen and run once per investigator | Every v2 choice was made on DEV; TEST numbers are unseen by construction |
| 20 scenarios per planted type (7 types, 140 planted) + 15 clean and 15 noisy control weeks | Enough per type for 95% intervals that mean something; severities cycle small / medium / large |
| New chained type **delay_then_reviews**: deliveries of one seller_state delayed in week W, reviews of that seller_state's orders drop in week W+1 (the investigated week, metric avg_review_score); truth = that seller_state, rate | Tests the "downstream effect" expectation (delay then reviews) |
| TEST target weeks never overlap DEV weeks and come from the same pool as DEV (from 2017-05-01, same excluded event weeks), each with at least 8 weeks of prior data: 8 distinct weeks | A first TEST build also used Olist's launch weeks (2017-03-27 to 2017-04-24) to find more non-DEV weeks. Diagnosed after the first 23 v2 runs: those weeks have only 12 to 16 weeks of ramp-up history and tiny segments, so even the statistical baseline B2 found 4 of 51 planted causes there vs 47 of 89 in the standard-pool weeks. That was a test-set construction error, not a property of any investigator; the set was rebuilt with the DEV pool rule and all TEST runs restarted. The discarded runs are kept in `results/v2/archive_launch_weeks/` |
| Target segments drawn by rule from the largest segments of each dimension (mix shifts: large low-AOV categories), so most TEST segments never appeared on DEV | A held-out test of the method, not of memorised segments |
| A planted scenario must change its target metric; otherwise its week is redrawn (1 case) | Data validity: an injection that touches no rows would be impossible to solve |
| No recalibration: the segment-scan threshold stays 4.25 | Thresholds are set on natural weeks only |
| Headline intervals are **week-clustered bootstrap** 95% CIs (resample the 8 target weeks with replacement, 2,000 draws) for top-1, top-3 and F1; Wilson intervals are also reported | Scenarios sharing a week share its data, so they are correlated; treating them as independent (Wilson) would give intervals that are too narrow |
| Stability: v2 rerun on 50 random TEST scenarios with the gateway cache bypassed; cache rerun on 30 other random scenarios | Measures run-to-run agreement and cache behaviour on the same code |
| Parallel workers (`--workers N`) tried on TEST and dropped back to 1 | With 2 workers NVIDIA returned 99 rate-limit errors in 15 minutes for 5 scenarios: the free tier limits Ultra per model per minute, so concurrency cannot raise throughput. A budget guard in the TEST orchestrator detected the spike and dropped to 1 worker automatically; the option stays for providers with more headroom (a unit test checks workers=1 and workers=2 give identical rows) |

## Model and provider

| Decision | Why |
|---|---|
| Headline model: **`nvidia/nemotron-3-ultra-550b-a55b`** on the NVIDIA API catalog free tier (build.nvidia.com), through the gateway | The only free, card-free option that worked for a full run (below) |
| Gemini Flash dropped | Its free tier allows only 20 requests per day and 5 per minute (seen in AI Studio); one evaluation needs hundreds of calls |
| Mistral dropped | Mistral's free API returned a 0 requests/min limit (`x-ratelimit-limit-req-minute: 0`) for every model on the account, even with a fresh key; adding a card was not possible |
| Why Nemotron 3 Ultra | One tool-calling request each: Kimi K3 gave no answer in 120 s; DeepSeek V4.1 Flash took 38 s (over the gateway's then 30 s timeout) and returned thinking text; Nemotron 3 Ultra returned proper tool calls in 2 s with thinking off |
| Thinking off (`chat_template_kwargs: {"enable_thinking": false}`) | Nemotron 3 thinks by default (`reasoning_content`); off gives plain content, fewer tokens and faster calls. The analysis is done by the tools, not by the model's reasoning |
| 9 requests per minute for the nvidia preset | NVIDIA allows about 40 per minute, but the gateway limits each client key to about 10 per minute |
| Gateway `REQUEST_TIMEOUT_S` raised from 30 to 120 s | Large free-tier models can take over 30 s on long prompts; a timeout would only cause retries |
| Local Qwen 2.5 (Ollama) kept as the optional comparison | Runs on the laptop for free; slow on CPU, so it may be run on a labelled subset |
