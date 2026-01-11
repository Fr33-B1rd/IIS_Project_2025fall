#!/usr/bin/env python
"""
main.py (grounded + emotion-during-listening)

- Furhat Realtime API
- EmotionDetector (webcam, emotion sampled DURING ASR listening window)
- StoryManager (A Distressed Damsel)
- NarrativeEngine (Gemini, story-grounded)

This version avoids custom frame-buffer logic and uses EmotionDetector.start_buffering/get_emotion_from_buffer.
"""

import logging
from typing import Optional, Tuple
import threading
import time

from furhat_realtime_api import FurhatClient

from perception import EmotionDetector
from emotion_state import EmotionMeter
from story_manager import StoryManager
from narrative import NarrativeEngine
from visualization import EmotionVisualizer


def run_application(
    robot_ip: str = "127.0.0.1",
    api_key: Optional[str] = None,
    test_mode: bool = False,
    show_ui: bool = True,
):
    # 1) Connect to Furhat
    try:
        furhat = FurhatClient(robot_ip, api_key) if api_key else FurhatClient(robot_ip)
        furhat.set_logging_level(logging.INFO)
        furhat.connect()
        print(f"[MAIN] Connected to Furhat Realtime API at {robot_ip}")
        furhat.request_voice_config(name="Matthew", input_language=True)
    except Exception as e:
        print(f"[CRITICAL] Could not connect to Furhat Realtime API: {e}")
        return

    # 2) Init modules (keep args compatible with your current perception.py signature)
    try:
        eyes = EmotionDetector(
            model_path="dinov3_svm_3class.joblib",
            camera_index=0,
            frames_per_call=5,
            pmax_threshold=0.45,
            margin_threshold=0.10,
        )
    except Exception as e:
        print(f"[CRITICAL] Could not start EmotionDetector: {e}")
        try:
            furhat.disconnect()
        except Exception:
            pass
        return

    brain = NarrativeEngine(test_mode=test_mode)
    meter = EmotionMeter()
    story = StoryManager(script_path="A_Distressed_Damsel.PDF")

    mode_str = "TEST MODE" if test_mode else "NORMAL DM MODE"
    print("--- Emotion-Adaptive Dungeon Master (Realtime API) ---")
    print(f"--- Running in {mode_str} ---")

    # Visualization state (optional)
    viz = None
    viz_stop = threading.Event()
    viz_lock = threading.Lock()
    viz_state = {
        "label": "uncertain",
        "level": "low",
        "meters": {"excited": 0.0, "nervous": 0.0, "confused": 0.0},
        "strategy": "Waiting for emotion update...",
        "deep_help": False,
    }

    def _extract_visual_state(decision_obj) -> Tuple[str, str, dict, bool]:
        if hasattr(decision_obj, "dominant"):
            label = str(getattr(decision_obj, "dominant", "uncertain"))
            level = str(getattr(decision_obj, "level", "low"))
            meters = dict(getattr(decision_obj, "meters", {}))
            deep_help = bool(getattr(decision_obj, "deep_help_mode", False))
            return label, level, meters, deep_help
        if isinstance(decision_obj, tuple) and len(decision_obj) >= 1:
            label = str(decision_obj[0])
            return label, "low", {}, False
        if isinstance(decision_obj, str):
            return decision_obj, "low", {}, False
        return "uncertain", "low", {}, False

    def _viz_loop():
        if viz is None:
            return
        while not viz_stop.is_set():
            frame = eyes.get_latest_frame()
            with viz_lock:
                state = dict(viz_state)
            quit_now = viz.update(
                frame_bgr=frame,
                label=state["label"],
                level=state["level"],
                meters=state["meters"],
                strategy_text=state["strategy"],
                deep_help_mode=state["deep_help"],
            )
            if quit_now:
                viz_stop.set()
                break
            time.sleep(0.05)
        if viz is not None:
            viz.close()

    if show_ui:
        try:
            viz = EmotionVisualizer()
            threading.Thread(target=_viz_loop, daemon=True).start()
        except Exception as e:
            print(f"[WARNING] Could not start visualization UI: {e}")
            viz = None

    # 3) Story-specific opening (prevents generic hallucinations)
    try:
        if test_mode:
            greeting = "This is emotion test mode. Ask: what's my emotion right now, and how should you react?"
        else:
            greeting = (
                "As you travel, someone crashes through the woods: a barefoot woman, torn and bloody, pleading, "
                "Please help me—my son is missing!"
            )
        furhat.request_speak_text(greeting, wait=True, abort=True)
    except Exception as e:
        print(f"[WARNING] Failed to speak greeting: {e}")

    try:
        while True:
            # A) Listen (capture emotion DURING listening)
            print("[Furhat] Listening...")
            try:
                eyes.start_buffering(fps=8.0, max_frames=80)
            except Exception:
                # buffering is optional; if it fails we still listen
                pass

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

            # Stop buffering and compute emotion from the buffered frames
            try:
                emo_label, emo_probs = eyes.get_emotion_from_buffer()
            except Exception:
                # fallback to snapshot
                try:
                    emo_label, emo_probs = eyes.get_emotion_snapshot()
                except Exception:
                    emo_label, emo_probs = "uncertain", {}

            user_text = (user_text or "").strip()
            print(f"[User] {user_text}")
            print(f"[Perception] User emotion: {(emo_label, emo_probs)}")

            if not user_text:
                furhat.request_speak_text(
                    "I did not quite catch that. Could you say it again?",
                    wait=True,
                    abort=True,
                )
                continue

            lowered = user_text.lower()
            if any(w in lowered for w in ["quit", "exit", "stop game", "stop"]):
                furhat.request_speak_text("Our story ends here, for now. Farewell.", wait=True, abort=True)
                break

            try:
                meter.apply_user_recovery(user_text)
                decision = meter.update(prob_dict=emo_probs, label=emo_label)
            except Exception:
                decision = (emo_label, emo_probs)

            # B) Story context + DM response
            story_ctx = story.get_context(user_text)
            response_text, response_gesture = brain.generate_response(
                user_text, decision, story_context=story_ctx
            )
            print(f"[DM] Text: {response_text} | Gesture: {response_gesture}")

            # C) Act
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
                print(f"[Warning] Could not speak text: {e}")

            try:
                label, level, meters_map, deep_help = _extract_visual_state(decision)
                strategy = brain._strategy_for_emotion(label, level=level, deep_help_mode=deep_help)
                with viz_lock:
                    viz_state["label"] = label
                    viz_state["level"] = level
                    if meters_map:
                        viz_state["meters"] = meters_map
                    viz_state["strategy"] = f"I sense the user is {label} ({level}). {strategy}"
                    viz_state["deep_help"] = deep_help
            except Exception:
                pass

            try:
                used_deep_help = bool(getattr(decision, "deep_help_mode", False))
                meter.apply_post_dm_action(used_deep_help=used_deep_help)
            except Exception:
                pass

    except KeyboardInterrupt:
        print("[MAIN] Interrupted by user.")
    finally:
        try:
            eyes.close()
        except Exception:
            pass
        try:
            viz_stop.set()
        except Exception:
            pass
        try:
            furhat.disconnect()
        except Exception:
            pass
        print("[MAIN] Shutting down TRPG DM.")


if __name__ == "__main__":
    run_application(robot_ip="127.0.0.1", test_mode=False)
