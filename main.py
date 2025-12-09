#!/usr/bin/env python
"""
main.py (Realtime API version)

- Uses Furhat Realtime API (synchronous FurhatClient)
- Uses EmotionDetector (webcam + DINOv3 + SVM + smoothing)
- Uses NarrativeEngine (Gemini-based Dungeon Master)
"""

import logging
import time

import cv2
from furhat_realtime_api import FurhatClient

from perception import EmotionDetector
from narrative import NarrativeEngine


def run_application(
    robot_ip: str = "127.0.0.1",
    api_key: str | None = None,
):
    """
    Start the emotion-adaptive TRPG DM application.
    """
    # 1. Connect to Furhat Realtime API
    try:
        if api_key:
            furhat = FurhatClient(robot_ip, api_key)
        else:
            furhat = FurhatClient(robot_ip)

        furhat.set_logging_level(logging.INFO)
        furhat.connect()
        print(f"[MAIN] Connected to Furhat Realtime API at {robot_ip}")

        furhat.request_voice_config(name="Matthew", input_language=True)
        print("[MAIN] Voice configured.")

    except Exception as e:
        print(f"[CRITICAL] Could not connect to Furhat Realtime API: {e}")
        return

    # 2. Init Eyes (emotion) and Brain (narrative)
    try:
        eyes = EmotionDetector(
            model_path="dinov3_svm_3class.joblib",
            smooth_window=15,
            camera_index=0,
            frames_per_call=5,
        )
    except Exception as e:
        print(f"[CRITICAL] Could not start EmotionDetector: {e}")
        furhat.disconnect()
        return

    brain = NarrativeEngine()

    print("--- Emotion-Adaptive Dungeon Master (Realtime API) ---")

    # Initial greeting
    try:
        furhat.request_speak_text(
            "Welcome, adventurer. I am your Dungeon Master. When you are ready, tell me what you do.",
            wait=True,
            abort=True,
        )
    except Exception as e:
        print(f"[WARNING] Failed to speak greeting: {e}")

    try:
        while True:
            # A. PERCEPTION: get emotional state + preview frame
            current_emotion, preview_frame = eyes.get_emotion(return_frame=True)
            print(f"[Perception] User emotion: {current_emotion}")

            # Show preview frame if available
            if preview_frame is not None:
                vis = preview_frame.copy()
                cv2.putText(
                    vis,
                    f"{current_emotion}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("EmotionPerception (Smoothed)", vis)

                # This keeps the window responsive; it won't auto-close
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    # Optional: allow quitting the whole app via 'q'
                    print("[MAIN] 'q' pressed in preview window, exiting.")
                    break

            # B. LISTENING: ASR via Realtime API (blocking until user speaks or timeout)
            print("[Furhat] Listening...")
            try:
                user_text = furhat.request_listen_start(
                    partial=False,
                    concat=True,
                    stop_no_speech=True,
                    stop_robot_start=True,
                    stop_user_end=True,
                    resume_robot_end=False,
                    no_speech_timeout=8.0,
                    end_speech_timeout=1.0,
                )
            except Exception as e:
                print(f"[WARNING] Listening failed: {e}")
                user_text = ""

            user_text = (user_text or "").strip()
            print(f"[User] {user_text}")

            if not user_text:
                furhat.request_speak_text(
                    "I did not quite catch that. Could you say it again?",
                    wait=True,
                    abort=True,
                )
                continue

            # Optional: exit keywords
            lowered = user_text.lower()
            if any(w in lowered for w in ["quit", "exit", "stop game", "stop"]):
                furhat.request_speak_text(
                    "Our adventure ends here, for now. Farewell, brave soul.",
                    wait=True,
                    abort=True,
                )
                break

            # C. REASONING: ask Gemini DM
            response_text, response_gesture = brain.generate_response(user_text, current_emotion)
            print(f"[DM] Text: {response_text} | Gesture: {response_gesture}")

            # D. ACTION: gesture + speech
            try:
                if response_gesture:
                    furhat.request_gesture_start(
                        name=response_gesture,
                        intensity=1.0,
                        duration=1.0,
                        wait=False,
                    )
            except Exception as e:
                print(f"[Warning] Could not trigger gesture {response_gesture}: {e}")

            try:
                furhat.request_speak_text(response_text, wait=True, abort=True)
            except Exception as e:
                print(f"[Warning] Could not speak response: {e}")

            # E. Update story state
            brain.advance_story()

            time.sleep(0.3)

    finally:
        try:
            eyes.close()
        except Exception:
            pass

        try:
            furhat.disconnect()
        except Exception:
            pass

        print("[MAIN] Shutting down TRPG DM.")


if __name__ == "__main__":
    run_application(robot_ip="127.0.0.1")
