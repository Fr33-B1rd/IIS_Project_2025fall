# narrative.py
import os
from typing import Any, Dict, Optional, Tuple

from google import genai
from google.genai import types


class NarrativeEngine:
    """Gemini-based Dungeon Master (emotion-adaptive, story-grounded)."""

    def __init__(self, test_mode: bool = False):
        self.client = None
        self.model_name = "gemini-3.0-flash"
        self.test_mode = test_mode

        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            print("WARNING: GEMINI_API_KEY not set. Gemini will be offline.")
        else:
            try:
                self.client = genai.Client(api_key=api_key)
                print("SUCCESS: Connected to Gemini Client.")
            except Exception as e:
                print(f"CRITICAL ERROR: Could not start Gemini Client. Details: {e}")
                self.client = None

    def _strategy_for_emotion(self, emotion_lower: str, level: str = "low", deep_help_mode: bool = False) -> str:
        if emotion_lower == "excited":
            if level == "high":
                return (
                    "Because you seem highly excited, I should keep the pace high, heighten drama, "
                    "and offer bold, vivid choices within canon."
                )
            return (
                "Because you seem excited, I should keep the pace high, raise the stakes, "
                "and offer bold dramatic choices."
            )
        if emotion_lower == "nervous":
            if level == "high":
                return (
                    "Because you seem very nervous, I should slow down, reassure you, "
                    "and offer the safest, clearest options."
                )
            return (
                "Because you seem nervous, I should slow down, give clearer guidance, "
                "and offer reassurance or safer options."
            )
        if emotion_lower == "confused":
            if level == "high" or deep_help_mode:
                return (
                    "Because you seem very confused, I should recap key facts, give step-by-step guidance, "
                    "and offer a simple next step."
                )
            return (
                "Because you seem confused, I should explain more clearly, repeat key info, "
                "and offer hints."
            )
        return "I'm not fully sure, so I should respond neutrally and check if you need clarification."

    def _gesture_for_emotion(self, emotion_lower: str) -> str:
        if emotion_lower == "excited":
            return "Surprise"
        if emotion_lower == "nervous":
            return "ExpressSad"
        return "Nod"

    @staticmethod
    def _normalize_emotion(
        emotion: Any,
    ) -> Tuple[str, Optional[Dict[str, float]], Optional[Dict[str, Any]], Optional[str], bool]:
        """Return (label_lower, prob_dict, cues, level, deep_help_mode) from various input shapes."""
        label = ""
        probs = None
        cues = None
        level = None
        deep_help_mode = False

        if emotion is None:
            label = ""
        elif isinstance(emotion, str):
            label = emotion
        elif hasattr(emotion, "dominant"):
            label = str(getattr(emotion, "dominant", ""))
            level = getattr(emotion, "level", None)
            deep_help_mode = bool(getattr(emotion, "deep_help_mode", False))
        elif isinstance(emotion, tuple) and len(emotion) >= 1:
            label = emotion[0] if isinstance(emotion[0], str) else str(emotion[0])
            if len(emotion) >= 2 and isinstance(emotion[1], dict):
                probs = emotion[1]
            if len(emotion) >= 3 and isinstance(emotion[2], dict):
                cues = emotion[2]
        elif isinstance(emotion, dict):
            label = str(emotion.get("dominant") or emotion.get("emotion") or emotion.get("label") or "")
            if isinstance(emotion.get("prob_dict"), dict):
                probs = emotion.get("prob_dict")
            if isinstance(emotion.get("cues"), dict):
                cues = emotion.get("cues")
            if isinstance(emotion.get("level"), str):
                level = emotion.get("level")
            if emotion.get("deep_help_mode") is not None:
                deep_help_mode = bool(emotion.get("deep_help_mode"))
        else:
            label = str(emotion)

        emotion_lower = (label or "").strip().lower()
        if not emotion_lower:
            emotion_lower = "confused"
        return emotion_lower, probs, cues, level, deep_help_mode

    @staticmethod
    def _style_for_emotion(emotion_lower: str, level: str, deep_help_mode: bool) -> str:
        if emotion_lower == "nervous":
            if level == "high":
                return "Player is very nervous: slow pace, reassure, give safest option, ask a gentle check-in."
            if level == "medium":
                return "Player is nervous: reassure, clarify steps, offer a safe option."
            return "Player is slightly nervous: calm, clear, and steady pacing."
        if emotion_lower == "excited":
            if level == "high":
                return "Player is very excited: energetic, vivid, raise stakes slightly (within canon)."
            if level == "medium":
                return "Player is excited: brisk pace, lively tone, offer bold choices."
            return "Player is mildly excited: upbeat, but keep it concise."
        if emotion_lower == "confused":
            if level == "high" or deep_help_mode:
                return "Player is very confused: recap facts, step-by-step guidance, 1-2 simple options."
            if level == "medium":
                return "Player is confused: clarify, summarize options, ask a simple next-step question."
            return "Player is slightly confused: clarify and offer a hint."
        return "Neutral fantasy narration."

    @staticmethod
    def _format_story_block(story_context: Any) -> str:
        if story_context is None:
            return ""
        try:
            beat_title = getattr(story_context, "beat_title", None) or story_context.get("beat_title")
            beat_summary = getattr(story_context, "beat_summary", None) or story_context.get("beat_summary")
            passages = getattr(story_context, "retrieved_passages", None) or story_context.get("retrieved_passages", [])
            beat_id = getattr(story_context, "beat_id", None) or story_context.get("beat_id")
            if passages:
                joined = "\n".join([f"- {p}" for p in passages])
            else:
                joined = "- (no passages retrieved)"
            return (
                f"[Beat ID]: {beat_id}\n"
                f"[Beat Title]: {beat_title}\n"
                f"[Beat Summary]: {beat_summary}\n"
                f"[Canon Passages]:\n{joined}\n"
            )
        except Exception:
            return ""

    def generate_response(self, user_input: str, emotion: Any, story_context: Any = None) -> Tuple[str, str]:
        emotion_lower, _, _, level, deep_help_mode = self._normalize_emotion(emotion)
        level = (level or "low").strip().lower()

        # TEST MODE
        if self.test_mode:
            tl = (user_input or "").lower()
            if ("what" in tl and "emotion" in tl) or ("how should you react" in tl) or ("how would you react" in tl):
                return (
                    f"I currently detect your emotion as {emotion_lower} ({level}). "
                    f"{self._strategy_for_emotion(emotion_lower, level=level, deep_help_mode=deep_help_mode)}",
                    self._gesture_for_emotion(emotion_lower),
                )

        # Offline fallback
        if self.client is None:
            fallback = (
                f"I sense you are {emotion_lower} ({level}). "
                f"{self._strategy_for_emotion(emotion_lower, level=level, deep_help_mode=deep_help_mode)}"
            )
            return fallback, self._gesture_for_emotion(emotion_lower)

        story_block = self._format_story_block(story_context)

        system_instruction = (
            "You are a tabletop RPG Dungeon Master running ONE specific micro-adventure.\n"
            "CRITICAL RULES:\n"
            "1) Use ONLY the provided Canon Passages + Beat Summary as story truth.\n"
            "2) Do NOT invent new locations, monsters, puzzles, or plotlines.\n"
            "3) If the player says something generic (e.g., 'let's start'), narrate the CURRENT beat.\n"
            "4) Keep output to 1–2 short sentences, plain text.\n"
        )

        style_guide = self._style_for_emotion(emotion_lower, level=level, deep_help_mode=deep_help_mode)

        prompt = (
            f"{story_block}\n"
            f"[Player Emotion]: {emotion_lower}\n"
            f"[Emotion Level]: {level}\n"
            f"[Deep Help Mode]: {deep_help_mode}\n"
            f"[Style]: {style_guide}\n"
            f"Player said: \"{user_input}\"\n"
            "Respond as DM for the CURRENT beat. If the player asks what to do, give 2-3 concrete options.\n"
            "Remember: no invented content outside canon.\n"
        )

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.3,
                ),
            )
            text = (response.text or "").strip()
            if not text:
                raise ValueError("Empty response from Gemini.")
        except Exception as e:
            print(f"Gemini Runtime Error: {e}")
            text = "I pause, considering. Tell me what you do next."

        return text, self._gesture_for_emotion(emotion_lower)
