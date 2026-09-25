"""OpenAI-compatible chat client. By default it talks to the LLM Gateway (Project 1).

Presets (python -m eval.run --model <name>):
  nvidia        : gateway with X-Provider-Force: nvidia -> nvidia/nemotron-3-ultra-550b-a55b (headline;
                  NVIDIA API catalog free tier, thinking off)
  mistral       : gateway with X-Provider-Force: mistral -> Mistral Small 4 (mistral-small-2603)
  mistral-medium: same provider, mistral-medium-latest
  gemini : gateway with X-Provider-Force: gemini -> Gemini Flash (no fallback, so no mixed-model runs)
  ollama : gateway with X-Provider-Force: ollama -> local Qwen 2.5
  gemini with DIRECT_GEMINI=true : straight to Gemini's OpenAI-compatible endpoint (no gateway)

The model name sent to the gateway matters: the gateway's cache key includes it, so runs of different
models never share cached answers.
"""
from __future__ import annotations

import json
import os
import time
from collections import deque
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()

GEMINI_OPENAI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Paid list prices, USD per 1M tokens (input, output). Used only for the ESTIMATED cost column:
# the free tiers we actually run on cost 0. The NVIDIA API catalog free tier has no list price, so nvidia
# runs report $0 and the token counts instead.
#   gemini : Gemini Flash, ai.google.dev/gemini-api/docs/pricing (checked 2026-09-24)
#   mistral       : Mistral Small 4 (mistral-small-2603), docs.mistral.ai/models/mistral-small-4-0-26-03 (checked 2026-09-25)
#   mistral-medium: Mistral Medium 3.5, docs.mistral.ai/models/mistral-medium-3-5-26-04 (checked 2026-09-25)
PRICES = {"gemini": (0.75, 3.75), "mistral": (0.15, 0.6), "mistral-medium": (1.5, 7.5), "nvidia": (0.0, 0.0),
          "nvidia-super": (0.0, 0.0)}
# exact model names: the Mistral free plan only has limits for these exact names
MISTRAL_MODELS = {"mistral": "mistral-small-2603", "mistral-medium": "mistral-medium-latest"}
MISTRAL_PRESETS = set(MISTRAL_MODELS)
# pinned exact name; chosen after a one-request function-calling test (see DECISIONS.md)
NVIDIA_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"
# smaller, much faster member of the same family; NVIDIA's rate limit is per model, so it is not throttled
# by Ultra's traffic (see DECISIONS.md)
NVIDIA_SUPER_MODEL = "nvidia/nemotron-3-super-120b-a12b"
# Nemotron 3 thinks by default (reasoning_content); off gives plain content and proper tool_calls
NVIDIA_EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": False}}


@dataclass
class LLMResponse:
    message: dict            # {"role": "assistant", "content": ..., "tool_calls": [...]}
    usage: dict = field(default_factory=dict)
    latency_s: float = 0.0
    cache: str | None = None     # gateway X-Cache header
    provider: str | None = None  # gateway X-Provider header
    served_model: str | None = None  # model name in the provider's response body


class Throttle:
    def __init__(self, per_min: float):
        self.gap = 60.0 / per_min if per_min > 0 else 0.0
        self.last = 0.0

    def wait(self):
        if self.gap:
            d = self.last + self.gap - time.time()
            if d > 0:
                time.sleep(d)
        self.last = time.time()


class TokenThrottle:
    """Keep (prompt + completion) tokens in any 60-second window under a budget (Mistral free plan: 20K/min).

    Before a call: wait until tokens used in the last 60 s plus this request's estimated size fit the budget.
    After a call: record its real usage (0 for a gateway cache hit, which never reaches Mistral)."""

    CHARS_PER_TOKEN = 2.4   # measured on a stored run (JSON-heavy prompts); low = cautious estimate
    EST_COMPLETION = 400    # tokens reserved for the reply

    def __init__(self, tokens_per_min: float, clock=time.time, sleep=time.sleep):
        self.budget = tokens_per_min
        self.window: deque = deque()   # (timestamp, tokens)
        self.clock, self.sleep = clock, sleep

    def estimate(self, messages, tools) -> int:
        chars = len(json.dumps(messages, default=str)) + len(json.dumps(tools, default=str))
        return int(chars / self.CHARS_PER_TOKEN) + self.EST_COMPLETION

    def used(self) -> int:
        now = self.clock()
        while self.window and now - self.window[0][0] >= 60:
            self.window.popleft()
        return sum(t for _, t in self.window)

    def wait(self, est: int) -> float:
        waited = 0.0
        # a request larger than the whole budget is sent once the window is empty
        while self.used() + est > self.budget and self.window:  # used() first: it drops expired calls
            d = max(0.5, 60 - (self.clock() - self.window[0][0]))
            self.sleep(d)
            waited += d
        return waited

    def record(self, tokens: int) -> None:
        self.window.append((self.clock(), tokens))


class ChatClient:
    def __init__(self, preset: str = "gemini"):
        from openai import OpenAI

        self.preset = preset
        self.extra_body: dict | None = None
        headers = {}
        direct = os.getenv("DIRECT_GEMINI", "false").lower() == "true"
        if preset == "gemini" and direct:
            base, key = GEMINI_OPENAI_URL, os.getenv("GEMINI_API_KEY", "")
            self.model = os.getenv("LLM_MODEL", "gemini-3.6-flash")
        elif preset == "ollama" and os.getenv("DIRECT_OLLAMA", "false").lower() == "true":
            base, key = os.getenv("OLLAMA_URL", "http://localhost:11434/v1"), "ollama"
            self.model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
        else:
            base = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
            key = os.getenv("LLM_API_KEY", "dev-key-1")
            if preset == "ollama":
                headers["X-Provider-Force"] = "ollama"
                self.model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
            elif preset in ("nvidia", "nvidia-super"):
                headers["X-Provider-Force"] = "nvidia"
                self.model = NVIDIA_MODEL if preset == "nvidia" else NVIDIA_SUPER_MODEL  # pinned; recorded per row
                self.extra_body = NVIDIA_EXTRA_BODY
            elif preset in MISTRAL_PRESETS:
                headers["X-Provider-Force"] = "mistral"
                self.model = MISTRAL_MODELS[preset]  # pinned; recorded in every run row
            else:
                # no fallback: a Gemini run must never be answered by another provider
                headers["X-Provider-Force"] = "gemini"
                self.model = os.getenv("LLM_MODEL", "gemini-3.6-flash")
        if os.getenv("LLM_CACHE_BYPASS", "false").lower() == "true":
            headers["X-Cache-Bypass"] = "true"  # stability repeats must reach the model, not the gateway cache
        self.client = OpenAI(base_url=base, api_key=key, default_headers=headers, timeout=120, max_retries=0)
        # Mistral's free plan allows about 1 request/s, but the gateway limits each client key to a 10-request
        # burst then ~10/min (REFILL_PER_SEC=0.1667), so 9/min stays under the gateway limit
        mistral = preset in MISTRAL_PRESETS
        if mistral:
            rpm = os.getenv("MISTRAL_REQUESTS_PER_MIN", "9")
        elif preset == "nvidia":
            # NVIDIA returned frequent 429s at 20/min and at 12/min (about 5 calls/min got through), so 6/min
            rpm = os.getenv("NVIDIA_REQUESTS_PER_MIN", "6")
        elif preset == "nvidia-super":
            # no 429s at 12/min in a burst test; the gateway's per-key limit (~24/min) is shared with other runs
            rpm = os.getenv("NVIDIA_SUPER_REQUESTS_PER_MIN", "16")
        else:
            rpm = os.getenv("REQUESTS_PER_MIN", "8")
        self.throttle = Throttle(float(rpm))
        # the binding Mistral free-plan limit is 20,000 tokens/min; 18,000 keeps a 10% margin
        self.token_throttle = TokenThrottle(float(os.getenv("MISTRAL_TPM_BUDGET", "18000"))) if mistral else None
        self.price = PRICES.get(preset, PRICES["gemini"])

    def chat(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        from openai import APIConnectionError, APIStatusError, APITimeoutError

        delay = 5.0
        est = self.token_throttle.estimate(messages, tools) if self.token_throttle else 0
        for attempt in range(8):
            if self.token_throttle:
                self.token_throttle.wait(est)
            self.throttle.wait()
            t0 = time.time()
            try:
                raw = self.client.chat.completions.with_raw_response.create(
                    model=self.model, messages=messages, tools=tools, tool_choice="auto", temperature=0,
                    extra_body=self.extra_body)
                resp = raw.parse()
                break
            except APIStatusError as e:
                if e.status_code in (429, 500, 502, 503, 504) and attempt < 7:
                    ra = e.response.headers.get("retry-after")
                    time.sleep(float(ra) if ra else delay)
                    delay = min(delay * 2, 90)
                    continue
                raise
            except (APIConnectionError, APITimeoutError):
                if attempt < 7:
                    time.sleep(delay)
                    delay = min(delay * 2, 90)
                    continue
                raise
        msg = resp.choices[0].message
        # keep the assistant message exactly as returned (including provider extras such as Gemini's
        # thought signatures on tool calls, which must be sent back unchanged in the next turn)
        m = msg.model_dump(exclude_none=True)
        m["role"] = "assistant"
        m.setdefault("content", None if m.get("tool_calls") else "")
        for i, tc in enumerate(m.get("tool_calls") or []):
            tc.setdefault("id", f"call_{i}")
            tc.setdefault("type", "function")
            tc["function"].setdefault("arguments", "{}")
        u = resp.usage
        usage = {"prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
                 "completion_tokens": getattr(u, "completion_tokens", 0) or 0} if u else {}
        cache = raw.headers.get("x-cache")
        if self.token_throttle:
            hit = bool(cache and cache.startswith("HIT"))
            self.token_throttle.record(0 if hit else (sum(usage.values()) or est))
        return LLMResponse(m, usage, time.time() - t0, cache, raw.headers.get("x-provider"),
                           getattr(resp, "model", None))


class ScriptedLLM:
    """Fake LLM for tests: returns pre-written assistant messages (or a function of the history)."""

    def __init__(self, script):
        self.script = list(script) if not callable(script) else script
        self.model = "scripted"
        self.calls = 0

    def chat(self, messages, tools) -> LLMResponse:
        self.calls += 1
        if callable(self.script):
            m = self.script(messages)
        else:
            m = self.script.pop(0)
        return LLMResponse(m, {"prompt_tokens": 100, "completion_tokens": 20}, 0.0, None, "scripted")
