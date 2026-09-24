"""Pydantic contracts for the final report (the agent submits it through the submit_report tool)."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class RootCause(BaseModel):
    rank: int = Field(..., ge=1, le=5)
    dimension: Literal["customer_state", "product_category", "seller_state", "seller_id",
                       "main_payment_type", "is_repeat_customer"]
    segment: str
    sub_segment: Optional[str] = Field(None, description="Optional deeper slice, e.g. 'product_category=electronics'")
    contribution_pct: Optional[float] = Field(None, description="Share of the total change explained, from drill_down")
    effect: Optional[Literal["rate", "mix", "volume"]] = Field(
        None, description="rate = behaviour inside the segment changed; mix = the segment's share changed; volume = order count changed (GMV/Orders)")
    p_value: Optional[float] = None
    impact_brl: Optional[float] = Field(None, description="R$ impact from estimate_impact, if the metric has one")
    confidence: Literal["high", "medium", "low"] = "medium"
    evidence_ids: list[str] = Field(default_factory=list)


class Report(BaseModel):
    metric: str
    week: str
    baseline: str = "prev 4-week avg"
    anomaly_confirmed: bool = Field(..., description="True only if the headline change is anomalous or at least one segment passed significance_test")
    change_pct: Optional[float] = None
    evidence_ids: list[str] = Field(default_factory=list)
    root_causes: list[RootCause] = Field(default_factory=list, description="Empty if no significant root cause was found")
    narrative: str = Field(..., description="3-5 plain-English sentences. Every number must come from a cited tool output.")
    recommended_actions: list[str] = Field(default_factory=list)
    charts: list[str] = Field(default_factory=list)
