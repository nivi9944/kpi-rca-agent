"""Shrink tool results before they go into the model's context, without losing information.

Only the copy the model reads is compacted. Evidence (inv.evidence) is stored in full, and the verifier
checks report numbers against that full copy, so grounding is unaffected.

Rules (numeric values are never removed):
  1. Echoed fields: a top-level field whose value equals the call argument of the same name
     (metric, week, dimension, segment, filters) is dropped, since the model just sent it.
  2. Repeated text: a top-level guidance string (method, rule, note, ...; 25+ characters) whose exact
     text was already shown earlier in this investigation is dropped. The first occurrence is kept.
  3. Compact JSON: no spaces after separators.
"""
from __future__ import annotations

import json

KEEP = {"evidence_id", "error"}
MIN_TEXT = 25  # only long guidance text is de-duplicated; short values such as "up" or a test name stay


class Compactor:
    """Per-investigation compactor; remembers which guidance strings the model has already seen."""

    def __init__(self):
        self.seen_text: set[tuple[str, str]] = set()

    def compact(self, out: dict, args: dict | None = None) -> dict:
        args = args or {}
        view = {}
        for k, v in out.items():
            if k not in KEEP:
                if k in args and v == args[k]:
                    continue  # rule 1: echo of the call's own argument
                if isinstance(v, str) and len(v) >= MIN_TEXT:
                    if (k, v) in self.seen_text:
                        continue  # rule 2: identical guidance text already shown
                    self.seen_text.add((k, v))
            view[k] = v
        return view


def dumps(obj) -> str:
    """JSON without spaces after separators; non-JSON values fall back to str()."""
    return json.dumps(obj, default=str, separators=(",", ":"))
