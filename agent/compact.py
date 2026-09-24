"""Compact tool results for the model's context, to cut tokens without losing information.

Only the text the MODEL reads is compacted. Evidence (inv.evidence) is stored in full, and the verifier
checks report numbers against that full evidence, so grounding is unaffected.

Rules (nothing numeric is ever removed):
  1. Echo-only fields: a top-level field whose value equals the argument of the same name that the model
     just sent (metric, week, dimension, segment, filters) is dropped. The model already has it.
  2. Duplicated text: a top-level guidance string (method, rule, note, ...; 25+ characters) whose exact text was already shown
     earlier in this investigation is dropped. The first occurrence is always kept, so no information is lost.
  3. Compact JSON: no spaces after separators.
"""
from __future__ import annotations

import json

KEEP = {"evidence_id", "error"}
MIN_TEXT = 25  # only long guidance text is de-duplicated; short values such as "up" or a test name always stay


class Compactor:
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
    return json.dumps(obj, default=str, separators=(",", ":"))
