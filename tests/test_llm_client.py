import json

import pytest

from agent.llm_client import PRICES, ChatClient, TokenThrottle
from agent.loop import AgentResult
from eval.run import row_spend, spend_by_file, worst_case_usd


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in ("DIRECT_GEMINI", "DIRECT_OLLAMA", "LLM_MODEL", "OLLAMA_MODEL", "REQUESTS_PER_MIN",
              "MISTRAL_REQUESTS_PER_MIN", "MISTRAL_TPM_BUDGET", "NVIDIA_REQUESTS_PER_MIN"):
        monkeypatch.delenv(k, raising=False)


@pytest.mark.parametrize("preset, provider, model", [
    ("nvidia", "nvidia", "nvidia/nemotron-3-ultra-550b-a55b"),
    ("mistral", "mistral", "mistral-small-2603"),
    ("mistral-medium", "mistral", "mistral-medium-latest"),
    ("gemini", "gemini", "gemini-3.6-flash"),
    ("ollama", "ollama", "qwen2.5:7b"),
])
def test_each_preset_is_forced_to_its_own_provider(preset, provider, model):
    c = ChatClient(preset)
    # forced: the gateway never answers one model's run with another provider
    assert c.client.default_headers["X-Provider-Force"] == provider
    # own, exact model name: the gateway cache key includes it, and the free plan only knows exact names
    assert c.model == model


def test_mistral_throttles_and_prices():
    c = ChatClient("mistral")
    assert c.throttle.gap == pytest.approx(60 / 9)  # 9 requests/min, under the gateway's ~10/min per key
    assert c.token_throttle.budget == 18000       # 10% under the free plan's 20,000 tokens/min
    assert c.price == PRICES["mistral"] == (0.15, 0.6)
    assert ChatClient("mistral-medium").price == (1.5, 7.5)
    assert ChatClient("gemini").price == (0.75, 3.75) and ChatClient("gemini").token_throttle is None


class FakeClock:
    def __init__(self):
        self.t = 1000.0
        self.slept = []

    def __call__(self):
        return self.t

    def sleep(self, d):
        self.slept.append(d)
        self.t += d


def test_token_throttle_waits_for_the_60s_window():
    clock = FakeClock()
    tt = TokenThrottle(18000, clock=clock, sleep=clock.sleep)
    assert tt.wait(12000) == 0            # empty window: send now
    tt.record(12000)
    clock.t += 10
    assert tt.wait(5000) == 0             # 12,000 + 5,000 fits under 18,000
    tt.record(5000)
    clock.t += 5
    waited = tt.wait(9000)                # 17,000 used: wait until the first call leaves the window
    assert waited == pytest.approx(45)
    assert tt.used() == 5000


def test_token_throttle_sends_an_oversized_request_once_the_window_is_empty():
    clock = FakeClock()
    tt = TokenThrottle(18000, clock=clock, sleep=clock.sleep)
    tt.record(3000)
    assert tt.wait(25000) == pytest.approx(60)
    assert tt.used() == 0


def test_estimated_cost_and_spend_use_the_model_price():
    r = AgentResult(report=None, verification=None, prompt_tokens=1_000_000, completion_tokens=100_000,
                    billable_prompt_tokens=400_000, price_per_m=PRICES["mistral-medium"])
    assert r.est_cost_usd() == pytest.approx(1.5 + 0.75)
    assert r.est_spend_usd() == pytest.approx(0.6)  # cache hits are not billable
    assert AgentResult(report=None, verification=None, prompt_tokens=1_000_000).est_cost_usd() == pytest.approx(0.75)


def test_budget_guard_totals_every_run_file(tmp_path):
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs_cache").mkdir()
    (tmp_path / "runs" / "mistral.jsonl").write_text(
        json.dumps({"id": "a", "est_spend_usd": 0.02, "est_cost_usd": 0.03}) + "\n"
        + json.dumps({"id": "b", "est_spend_usd": 0.05}) + "\n", encoding="utf-8")
    (tmp_path / "runs" / "gemini.jsonl").write_text(json.dumps({"id": "a", "est_cost_usd": 0.1}) + "\n",
                                                    encoding="utf-8")
    (tmp_path / "runs_cache" / "mistral.jsonl").write_text(json.dumps({"id": "a", "est_spend_usd": 0.0}) + "\n",
                                                           encoding="utf-8")
    spend = spend_by_file(tmp_path)
    assert spend == {"runs/gemini.jsonl": 0.1, "runs/mistral.jsonl": 0.07, "runs_cache/mistral.jsonl": 0.0}
    assert row_spend({"est_cost_usd": 0.03}) == 0.03  # older rows without est_spend_usd count their full cost
    assert worst_case_usd("mistral", tmp_path / "runs" / "mistral.jsonl") == 0.05
    # no rows yet: the whole token budget at list price
    assert worst_case_usd("mistral", tmp_path / "none.jsonl") == pytest.approx(250_000 * 0.15 / 1e6 + 10_000 * 0.6 / 1e6)


def test_nvidia_preset_is_free_thinking_off_and_under_the_gateway_limit(tmp_path):
    c = ChatClient("nvidia")
    assert c.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}
    assert ChatClient("gemini").extra_body is None
    assert c.throttle.gap == pytest.approx(60 / 9)  # gateway allows ~10/min per key
    assert c.price == (0.0, 0.0) and c.token_throttle is None
    assert worst_case_usd("nvidia", tmp_path / "none.jsonl") == 0.0  # the budget guard never triggers
