#!/usr/bin/env python
"""
test_perception.py

Simple webcam test for EmotionPerception with temporal smoothing.
Press 'q' to quit.
"""

import cv2
from perception import EmotionPerception


def main():
    # You can tweak smooth_window (e.g., 5, 10, 20)
    ep = EmotionPerception("dinov3_svm_3class.joblib", smooth_window=15)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0).")

    print("Webcam opened. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from webcam.")
            break

        # Smoothed prediction
        label, prob_dict = ep.predict(frame)
        top_prob = prob_dict.get(label, 0.0)

        text = f"{label} ({top_prob*100:.1f}%)"
        cv2.putText(
            frame,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("EmotionPerception (Smoothed)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
