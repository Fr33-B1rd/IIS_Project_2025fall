#!/usr/bin/env python
"""
Realtime emotion perception test.
Uses webcam + trained model and prints current prediction.
Press "q" to quit.
"""

from __future__ import annotations

import time

import cv2

from perception import EmotionDetector


def main():
    detector = EmotionDetector(
        model_path="dinov3_svm_3class.joblib",
        camera_index=0,
        frames_per_call=5,
        pmax_threshold=0.55,
        margin_threshold=0.15,
    )

    window_name = "Emotion Test"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            label, prob_dict = detector.get_emotion_snapshot()
            frame = detector.get_latest_frame()

            msg = f"{label} | " + ", ".join([f"{k}:{v:.2f}" for k, v in prob_dict.items()])
            print(msg)

            if frame is not None:
                cv2.putText(
                    frame,
                    label,
                    (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow(window_name, frame)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

            time.sleep(0.1)
    finally:
        detector.close()
        try:
            cv2.destroyWindow(window_name)
            cv2.destroyAllWindows()
            for _ in range(3):
                cv2.waitKey(1)
        except Exception:
            pass


if __name__ == "__main__":
    main()
