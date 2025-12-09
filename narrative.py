# narrative.py
from google import genai
from google.genai import types
import os


class NarrativeEngine:
    """
    Story / dialogue engine using Gemini.

    - Keeps a simple state machine (INTRO → PUZZLE → COMBAT, etc.)
    - Uses emotion + user text to generate a short DM-style response.
    """

    def __init__(self):
        self.current_state = "INTRO"
        self.client = None
        self.model_name = "gemini-2.5-flash"

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

    def generate_response(self, user_input: str, emotion: str):
        """
        Generates Furhat's response using Google Gemini.
        Args:
            user_input: player's utterance (plain text)
            emotion: "excited", "nervous", or "confused"
        Returns:
            (response_text, gesture_name)
        """

        # Safety check
        if self.client is None:
            print("Gemini is offline. Using fallback text.")
            return "I cannot hear the spirits right now. Please check my magic connection.", "ExpressSad"

        # System persona
        system_instruction = """
        You are the Dungeon Master for a tabletop RPG.
        Your role is to describe the scene, react to the player's actions, and advance the plot.
        Keep your responses SHORT (maximum 2 sentences) because they will be spoken by a robot.
        Do not use markdown formatting, only plain text.
        Stay in character as a fantasy Dungeon Master.
        """

        # Emotion-adaptive style guide
        emotion_lower = (emotion or "").strip().lower()
        style_guide = "Speak in a neutral, narrative tone."
        if emotion_lower == "nervous":
            style_guide = "The player is nervous. Use calming, reassuring language and offer a subtle hint."
        elif emotion_lower == "excited":
            style_guide = "The player is excited. Use fast-paced, dramatic, high-energy vocabulary and raise the stakes."
        elif emotion_lower == "confused":
            style_guide = "The player is confused. Speak slowly and clearly, explain the situation in simple words."

        prompt = f"""
        [Current Game State]: {self.current_state}
        [Player Emotion]: {emotion_lower}
        [Style Instruction]: {style_guide}
        Player said: "{user_input}"
        How do you respond? Keep it within 2 sentences.
        """

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.7,
                ),
            )
            text = (response.text or "").strip()
        except Exception as e:
            print(f"Gemini Runtime Error: {e}")
            text = "The spirits are strangely quiet. Let us take a small step forward and see what happens."

        # Map emotion to gesture name (Furhat gesture IDs depend on your setup)
        gesture = "Nod"
        if emotion_lower == "excited":
            gesture = "Surprise"
        elif emotion_lower == "nervous":
            gesture = "ExpressSad"

        return text, gesture

    def advance_story(self):
        """
        Simple finite-state machine for the TRPG progression.
        You can extend this with more states and logic later.
        """
        if self.current_state == "INTRO":
            self.current_state = "PUZZLE"
        elif self.current_state == "PUZZLE":
            self.current_state = "COMBAT"
        elif self.current_state == "COMBAT":
            self.current_state = "EPILOGUE"
        # else: keep EPILOGUE as an end state for now
