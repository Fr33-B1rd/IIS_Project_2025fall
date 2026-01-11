#!/usr/bin/env python
"""
visualization.py

OpenCV HUD for live camera + emotion meter + response strategy text.
"""

from __future__ import annotations

from typing import Dict, Optional

import cv2
import numpy as np


class EmotionVisualizer:
    def __init__(
        self,
        window_name: str = "Emotion HUD",
        cam_size: tuple = (480, 360),
        meter_size: tuple = (360, 360),
        text_height: int = 140,
    ):
        self.window_name = window_name
        self.cam_w, self.cam_h = cam_size
        self.meter_w, self.meter_h = meter_size
        self.text_h = int(text_height)

        self.total_w = self.cam_w + self.meter_w
        self.total_h = self.cam_h + self.text_h

    def _blank_frame(self, color=(30, 30, 30)) -> np.ndarray:
        canvas = np.zeros((self.total_h, self.total_w, 3), dtype=np.uint8)
        canvas[:] = color
        return canvas

    @staticmethod
    def _wrap_text(text: str, max_width: int, font, font_scale: float, thickness: int) -> list:
        words = (text or "").split()
        if not words:
            return [""]
        lines = []
        cur = words[0]
        for w in words[1:]:
            test = f"{cur} {w}"
            (tw, _), _ = cv2.getTextSize(test, font, font_scale, thickness)
            if tw <= max_width:
                cur = test
            else:
                lines.append(cur)
                cur = w
        lines.append(cur)
        return lines

    def _draw_meter(self, panel: np.ndarray, meters: Dict[str, float]) -> None:
        center = (self.meter_w // 2, self.meter_h // 2)
        radius = int(min(self.meter_w, self.meter_h) * 0.35)
        font = cv2.FONT_HERSHEY_SIMPLEX

        axes = {
            "nervous": -90,
            "excited": 30,
            "confused": 150,
        }

        # Axes and labels
        for label, deg in axes.items():
            rad = np.deg2rad(deg)
            x = int(center[0] + radius * np.cos(rad))
            y = int(center[1] + radius * np.sin(rad))
            cv2.line(panel, center, (x, y), (200, 200, 200), 1)
            lx = int(center[0] + (radius + 20) * np.cos(rad))
            ly = int(center[1] + (radius + 20) * np.sin(rad))
            cv2.putText(panel, label, (lx - 20, ly), font, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

        # Polygon for current meter values
        pts = []
        for label, deg in axes.items():
            v = float(meters.get(label, 0.0))
            v = 0.0 if v < 0.0 else 1.0 if v > 1.0 else v
            rad = np.deg2rad(deg)
            x = int(center[0] + radius * v * np.cos(rad))
            y = int(center[1] + radius * v * np.sin(rad))
            pts.append([x, y])
        pts = np.array(pts, dtype=np.int32)
        if len(pts) == 3:
            cv2.polylines(panel, [pts], isClosed=True, color=(120, 220, 120), thickness=2)
            cv2.fillPoly(panel, [pts], color=(60, 160, 60))

        cv2.circle(panel, center, 3, (255, 255, 255), -1)

    def update(
        self,
        frame_bgr: Optional[np.ndarray],
        label: str,
        level: str,
        meters: Dict[str, float],
        strategy_text: str,
        deep_help_mode: bool = False,
    ) -> bool:
        canvas = self._blank_frame()

        # Camera panel
        if frame_bgr is None:
            cam = np.zeros((self.cam_h, self.cam_w, 3), dtype=np.uint8)
            cam[:] = (20, 20, 20)
        else:
            cam = cv2.resize(frame_bgr, (self.cam_w, self.cam_h), interpolation=cv2.INTER_AREA)

        cv2.putText(
            cam,
            f"{label} ({level})",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        if deep_help_mode:
            cv2.putText(
                cam,
                "deep-help",
                (10, 56),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 200, 255),
                2,
                cv2.LINE_AA,
            )

        canvas[0:self.cam_h, 0:self.cam_w] = cam

        # Meter panel
        meter_panel = np.zeros((self.meter_h, self.meter_w, 3), dtype=np.uint8)
        meter_panel[:] = (25, 25, 25)
        self._draw_meter(meter_panel, meters)
        cv2.putText(
            meter_panel,
            "Emotion Meter",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )
        canvas[0:self.meter_h, self.cam_w:self.total_w] = meter_panel

        # Strategy text panel
        text_panel = np.zeros((self.text_h, self.total_w, 3), dtype=np.uint8)
        text_panel[:] = (15, 15, 15)
        font = cv2.FONT_HERSHEY_SIMPLEX
        lines = self._wrap_text(strategy_text, self.total_w - 20, font, 0.6, 1)
        y = 28
        for line in lines[:4]:
            cv2.putText(text_panel, line, (10, y), font, 0.6, (230, 230, 230), 1, cv2.LINE_AA)
            y += 22
        canvas[self.cam_h:self.total_h, 0:self.total_w] = text_panel

        cv2.imshow(self.window_name, canvas)
        key = cv2.waitKey(1) & 0xFF
        return key == ord("q")

    def close(self) -> None:
        try:
            cv2.destroyWindow(self.window_name)
        except Exception:
            pass
