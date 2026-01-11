#!/usr/bin/env python
"""
emotion_state.py

Emotion meter + decision logic for the Furhat TRPG DM.

Goal:
- Keep a persistent, decaying record of user emotion over time (excited / nervous / confused)
- Provide derived "dominant emotion", tiered "level", and flags (e.g., deep_help_mode)
- Include recovery mechanisms so meters don't monotonically rise

Designed to be lightweight and demo-stable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import math
import time


EmotionName = str  # "excited" | "nervous" | "confused" | "uncertain" | "mixed"


@dataclass
class EmotionDecision:
    """What the dialogue engine should use."""
    dominant: EmotionName
    level: str                 # "low" | "medium" | "high"
    meters: Dict[EmotionName, float]
    margin: float
    deep_help_mode: bool
    mixed: bool
    uncertain: bool
    cues: Dict[str, float]


class EmotionMeter:
    """
    Time-decayed emotion meter.

    Key features (P2-ish):
    - time-based exponential decay (stable if loop timing varies)
    - probability-driven reinforcement
    - uncertainty gating (p_max / margin)
    - hysteresis to prevent flip-flopping
    - confusion streak + deep-help trigger
    - recovery events (user understanding keywords; post deep-help dampening)
    """

    def __init__(
        self,
        emotions: Tuple[EmotionName, ...] = ("excited", "nervous", "confused"),
        # decay model: M(t) = M0 * exp(-lambda * dt)
        half_life_sec: float = 18.0,
        gain: float = 0.22,  # reinforcement magnitude (per update, scaled by probs)
        # uncertainty gate
        pmax_threshold: float = 0.45,
        prob_margin_threshold: float = 0.10,
        # dominance / switching hysteresis
        switch_margin: float = 0.08,
        switch_confirm_updates: int = 2,
        # tier thresholds
        low_th: float = 0.30,
        high_th: float = 0.60,
        # deep help
        confused_high_th: float = 0.60,
        confused_streak_updates: int = 3,
        # recovery
        recovery_on_understood_mul: float = 0.60,
        recovery_after_deep_help_mul: float = 0.85,
    ):
        self.emotions = tuple(emotions)
        self.meters: Dict[EmotionName, float] = {e: 0.0 for e in self.emotions}

        self.half_life_sec = max(1e-3, float(half_life_sec))
        self.lambda_decay = math.log(2.0) / self.half_life_sec
        self.gain = float(gain)

        self.pmax_threshold = float(pmax_threshold)
        self.prob_margin_threshold = float(prob_margin_threshold)

        self.switch_margin = float(switch_margin)
        self.switch_confirm_updates = int(max(1, switch_confirm_updates))
        self._switch_candidate: Optional[EmotionName] = None
        self._switch_count: int = 0
        self._dominant: EmotionName = "mixed"

        self.low_th = float(low_th)
        self.high_th = float(high_th)

        self.confused_high_th = float(confused_high_th)
        self.confused_streak_updates = int(max(1, confused_streak_updates))
        self._confused_streak = 0

        self.recovery_on_understood_mul = float(recovery_on_understood_mul)
        self.recovery_after_deep_help_mul = float(recovery_after_deep_help_mul)

        self.last_ts: float = time.time()
        self.last_decision: Optional[EmotionDecision] = None

    # ------------------------- utilities -------------------------

    @staticmethod
    def _clamp01(x: float) -> float:
        return 0.0 if x < 0.0 else 1.0 if x > 1.0 else float(x)

    @staticmethod
    def _normalize_probs(probs: Dict[EmotionName, float], emotions: Tuple[EmotionName, ...]) -> Dict[EmotionName, float]:
        # Keep only known classes, fill missing with 0, then renormalize.
        vals = {e: float(probs.get(e, 0.0)) for e in emotions}
        s = sum(max(0.0, v) for v in vals.values())
        if s <= 1e-12:
            return {e: 0.0 for e in emotions}
        return {e: max(0.0, v) / s for e, v in vals.items()}

    def _apply_time_decay(self, now: float) -> None:
        dt = max(0.0, float(now - self.last_ts))
        if dt <= 0.0:
            return
        factor = math.exp(-self.lambda_decay * dt)
        for e in self.emotions:
            self.meters[e] *= factor

    def _tier(self, v: float) -> str:
        if v < self.low_th:
            return "low"
        if v < self.high_th:
            return "medium"
        return "high"

    # ------------------------- recovery hooks -------------------------

    def apply_user_recovery(self, user_text: str) -> None:
        """Heuristic: if user indicates understanding, reduce confused meter."""
        t = (user_text or "").strip().lower()
        if not t:
            return
        keywords = [
            "ok", "okay", "i see", "got it", "understood", "makes sense",
            "clear now", "that helps", "aha", "thanks",
            # common variants
            "alright", "all right", "fine", "sure",
        ]
        if any(k in t for k in keywords):
            if "confused" in self.meters:
                self.meters["confused"] *= self.recovery_on_understood_mul
            if "nervous" in self.meters:
                # mild calming effect often follows understanding
                self.meters["nervous"] *= (0.85 + 0.15 * self.recovery_on_understood_mul)

    def apply_post_dm_action(self, used_deep_help: bool) -> None:
        """If DM used deep-help explanation, gently reduce confusion so we don't get stuck."""
        if used_deep_help and "confused" in self.meters:
            self.meters["confused"] *= self.recovery_after_deep_help_mul

    # ------------------------- main update -------------------------

    def update(
        self,
        prob_dict: Dict[EmotionName, float],
        label: Optional[EmotionName] = None,
        cues: Optional[Dict[str, float]] = None,
        now: Optional[float] = None,
    ) -> EmotionDecision:
        """
        Update meters from current perception output.

        Args:
            prob_dict: class->prob (ideally already smoothed from perception)
            label: optional hard label (used for streak & tie-break)
            cues: optional extra signals (motion, brightness, etc.) for logging/prompting
            now: timestamp (seconds). If None, uses time.time()
        """
        if now is None:
            now = time.time()

        # 1) decay
        self._apply_time_decay(now)

        # 2) reinforcement from probs
        probs = self._normalize_probs(prob_dict or {}, self.emotions)
        for e in self.emotions:
            self.meters[e] = self._clamp01(self.meters[e] + self.gain * probs.get(e, 0.0))

        # 3) determine uncertainty from raw probs (not meters)
        # Note: probs already normalized over 3 classes
        p_sorted = sorted((probs[e], e) for e in self.emotions)
        pmax, emax = p_sorted[-1]
        p2 = p_sorted[-2][0] if len(p_sorted) >= 2 else 0.0
        prob_margin = float(pmax - p2)

        uncertain = (pmax < self.pmax_threshold) or (prob_margin < self.prob_margin_threshold)

        # 4) determine dominant from meters + margin gate
        m_sorted = sorted((self.meters[e], e) for e in self.emotions)
        mmax, m_emax = m_sorted[-1]
        m2 = m_sorted[-2][0] if len(m_sorted) >= 2 else 0.0
        meter_margin = float(mmax - m2)

        mixed = (meter_margin < self.prob_margin_threshold)

        # candidate dominant (pre-hysteresis)
        candidate = "uncertain" if uncertain else ("mixed" if mixed else m_emax)

        # 5) hysteresis switching
        if candidate in ("mixed", "uncertain"):
            self._dominant = candidate
            self._switch_candidate = None
            self._switch_count = 0
        else:
            # if currently a concrete emotion, require candidate to be confidently better
            if self._dominant in self.emotions:
                cur = self._dominant
                if candidate == cur:
                    self._switch_candidate = None
                    self._switch_count = 0
                else:
                    # only consider switching if meter advantage is enough
                    if self.meters[candidate] - self.meters[cur] >= self.switch_margin:
                        if self._switch_candidate == candidate:
                            self._switch_count += 1
                        else:
                            self._switch_candidate = candidate
                            self._switch_count = 1
                        if self._switch_count >= self.switch_confirm_updates:
                            self._dominant = candidate
                            self._switch_candidate = None
                            self._switch_count = 0
                    else:
                        # not enough evidence to switch
                        self._switch_candidate = None
                        self._switch_count = 0
            else:
                # current is mixed/uncertain; accept candidate immediately
                self._dominant = candidate
                self._switch_candidate = None
                self._switch_count = 0

        # 6) confusion streak for deep-help
        hard_label = (label or "").strip().lower() if label else ""
        if hard_label == "confused":
            self._confused_streak += 1
        else:
            # decay streak rather than reset, to avoid bouncing
            self._confused_streak = max(0, self._confused_streak - 1)

        deep_help_mode = (self.meters.get("confused", 0.0) >= self.confused_high_th) and (
            self._confused_streak >= self.confused_streak_updates
        )

        # 7) level from dominant meter (for mixed/uncertain use max meter)
        if self._dominant in self.emotions:
            level = self._tier(self.meters[self._dominant])
        else:
            level = self._tier(mmax)

        decision = EmotionDecision(
            dominant=self._dominant,
            level=level,
            meters={e: float(self.meters[e]) for e in self.emotions},
            margin=meter_margin,
            deep_help_mode=bool(deep_help_mode),
            mixed=bool(self._dominant == "mixed"),
            uncertain=bool(self._dominant == "uncertain"),
            cues=dict(cues or {}),
        )
        self.last_ts = float(now)
        self.last_decision = decision
        return decision
