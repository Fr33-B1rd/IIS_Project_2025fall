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
            model_path="dinov3_svm_3class_mydata.joblib",
            camera_index=0,
            frames_per_call=5,
            pmax_threshold=0.55,
            margin_threshold=0.15,
        )
    except Exception as e:
        print(f"[CRITICAL] Could not start EmotionDetector: {e}")
        try:
            furhat.disconnect()
        except Exception:
            pass
        return

    brain = NarrativeEngine()
    meter = EmotionMeter()
    story = StoryManager(script_path="A_Distressed_Damsel.PDF")

    print("--- Emotion-Adaptive Dungeon Master (Realtime API) ---")
    print("--- Running in NORMAL DM MODE ---")

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
    last_emo_label = "uncertain"
    last_emo_probs = {}
    last_decision = None

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

    def _update_hud_from_decision(decision_obj) -> None:
        label, level, meters_map, deep_help = _extract_visual_state(decision_obj)
        strategy = brain._strategy_for_emotion(label, level=level, deep_help_mode=deep_help)
        with viz_lock:
            viz_state["label"] = label
            viz_state["level"] = level
            if meters_map:
                viz_state["meters"] = meters_map
            viz_state["strategy"] = f"I sense the user is {label} ({level}). {strategy}"
            viz_state["deep_help"] = deep_help

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

    viz_thread = None
    if show_ui:
        try:
            viz = EmotionVisualizer()
            viz_thread = threading.Thread(target=_viz_loop, daemon=True)
            viz_thread.start()
        except Exception as e:
            print(f"[WARNING] Could not start visualization UI: {e}")
            viz = None

    # 3) Story-specific opening (prevents generic hallucinations)
    try:
        greeting = (
            "Hello adventurer. I'm your Dungeon Master for this journey. "
            "Are you ready to begin our adventure?"
        )
        try:
            eyes.start_buffering(fps=8.0, max_frames=80)
        except Exception:
            pass
        furhat.request_speak_text(greeting, wait=True, abort=True)
        try:
            last_emo_label, last_emo_probs = eyes.get_emotion_from_buffer()
        except Exception:
            try:
                last_emo_label, last_emo_probs = eyes.get_emotion_snapshot()
            except Exception:
                last_emo_label, last_emo_probs = "uncertain", {}
        print(f"[Perception] User emotion (during listening): {(last_emo_label, last_emo_probs)}")
        try:
            last_decision = meter.update(prob_dict=last_emo_probs, label=last_emo_label)
            _update_hud_from_decision(last_decision)
        except Exception:
            pass
    except Exception as e:
        print(f"[WARNING] Failed to speak greeting: {e}")

    try:
        while True:
            # A) Listen (capture emotion DURING listening)
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
                try:
                    eyes.start_buffering(fps=8.0, max_frames=80)
                except Exception:
                    pass
                furhat.request_speak_text(
                    "I did not quite catch that. Could you say it again?",
                    wait=True,
                    abort=True,
                )
                try:
                    last_emo_label, last_emo_probs = eyes.get_emotion_from_buffer()
                except Exception:
                    try:
                        last_emo_label, last_emo_probs = eyes.get_emotion_snapshot()
                    except Exception:
                        last_emo_label, last_emo_probs = "uncertain", {}
                print(f"[Perception] User emotion (during listening): {(last_emo_label, last_emo_probs)}")
                try:
                    last_decision = meter.update(prob_dict=last_emo_probs, label=last_emo_label)
                    _update_hud_from_decision(last_decision)
                except Exception:
                    pass
                continue

            lowered = user_text.lower()
            if any(w in lowered for w in ["quit", "exit", "stop game", "stop"]):
                try:
                    eyes.start_buffering(fps=8.0, max_frames=80)
                except Exception:
                    pass
                furhat.request_speak_text("Our story ends here, for now. Farewell.", wait=True, abort=True)
                try:
                    last_emo_label, last_emo_probs = eyes.get_emotion_from_buffer()
                except Exception:
                    try:
                        last_emo_label, last_emo_probs = eyes.get_emotion_snapshot()
                    except Exception:
                        last_emo_label, last_emo_probs = "uncertain", {}
                print(f"[Perception] User emotion (during listening): {(last_emo_label, last_emo_probs)}")
                try:
                    last_decision = meter.update(prob_dict=last_emo_probs, label=last_emo_label)
                    _update_hud_from_decision(last_decision)
                except Exception:
                    pass
                break

            try:
                meter.apply_user_recovery(user_text)
                decision = last_decision or (last_emo_label, last_emo_probs)
            except Exception:
                decision = (last_emo_label, last_emo_probs)

            # B) Story context + DM response
            dm_controls = brain.get_dm_controls(decision)
            story_ctx = story.get_context(user_text, dm_controls=dm_controls)
            response_text, response_gesture = brain.generate_response(
                user_text, decision, story_context=story_ctx
            )
            print(f"[DM] Text: {response_text} | Gesture: {response_gesture}")
            try:
                story.record_turn(user_text, response_text)
            except Exception:
                pass

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
                try:
                    eyes.start_buffering(fps=8.0, max_frames=80)
                except Exception:
                    pass
                furhat.request_speak_text(response_text, wait=True, abort=True)
                try:
                    last_emo_label, last_emo_probs = eyes.get_emotion_from_buffer()
                except Exception:
                    try:
                        last_emo_label, last_emo_probs = eyes.get_emotion_snapshot()
                    except Exception:
                        last_emo_label, last_emo_probs = "uncertain", {}
                print(f"[Perception] User emotion (during listening): {(last_emo_label, last_emo_probs)}")
                try:
                    last_decision = meter.update(prob_dict=last_emo_probs, label=last_emo_label)
                    _update_hud_from_decision(last_decision)
                except Exception:
                    pass
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
            if viz_thread is not None:
                viz_thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            furhat.disconnect()
        except Exception:
            pass
        print("[MAIN] Shutting down TRPG DM.")


if __name__ == "__main__":
    run_application(robot_ip="127.0.0.1")
