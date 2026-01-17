# story_manager.py
"""
StoryManager: loads a short TRPG adventure (PDF/text) and serves *small, relevant* story context
to the NarrativeEngine each turn.

Design goals (demo-friendly):
- Keep the DM "on-rails" with a small number of scenes ("beats") for predictable pacing.
- Provide retrieval over the script so Gemini can quote/ground details without dumping the whole PDF.
- Maintain a lightweight game state: current_scene, flags, discovered clues.

This module is intentionally system-agnostic: it does NOT enforce D&D rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re
import math
import time


def _normalize_ws(s: str) -> str:
    s = s.replace("\u00ad", "")  # soft hyphen
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _try_extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Best-effort PDF text extraction without heavy dependencies.
    Tries PyPDF2 first; if unavailable, raises RuntimeError.
    """
    try:
        import PyPDF2  # type: ignore
    except Exception as e:
        raise RuntimeError("PyPDF2 not available for PDF text extraction") from e

    text_parts: List[str] = []
    with pdf_path.open("rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            t = page.extract_text() or ""
            if t.strip():
                text_parts.append(t)
    return _normalize_ws("\n\n".join(text_parts))


def _split_into_chunks(text: str, max_chars: int = 800) -> List[str]:
    """
    Split text into reasonably small chunks to feed into prompts.
    Uses paragraph boundaries when possible.
    """
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[str] = []
    buf: List[str] = []
    cur = 0
    for p in paras:
        if cur + len(p) + 2 > max_chars and buf:
            chunks.append(_normalize_ws("\n\n".join(buf)))
            buf = [p]
            cur = len(p)
        else:
            buf.append(p)
            cur += len(p) + 2
    if buf:
        chunks.append(_normalize_ws("\n\n".join(buf)))
    return chunks


def _tokenize(s: str) -> List[str]:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    toks = [t for t in s.split() if len(t) >= 2]
    return toks


@dataclass
class StoryBeat:
    """A high-level scene/beat that keeps the demo paced and coherent."""
    beat_id: str
    title: str
    summary: str
    keywords: List[str] = field(default_factory=list)
    details: List[str] = field(default_factory=list)
    risk_low: List[str] = field(default_factory=list)
    risk_mid: List[str] = field(default_factory=list)
    risk_high: List[str] = field(default_factory=list)


@dataclass
class StoryContext:
    beat_id: str
    beat_title: str
    beat_summary: str
    retrieved_passages: List[str]
    flags: Dict[str, bool]
    recent_turns: List[Tuple[str, str]]
    beat_details: List[str]
    risk_cues: List[str]
    dm_controls: Dict[str, str]


class StoryManager:
    """
    Hybrid story manager:
    - Scene (beat) controller: 4–6 beats for a predictable demo.
    - Retrieval: select top-k chunks relevant to user_input and current beat.
    """

    def __init__(
        self,
        script_path: str,
        *,
        max_chunks: int = 3,
        max_chars_per_chunk: int = 800,
        max_recent_turns: int = 6,
    ):
        self.script_path = Path(script_path)
        self.max_chunks = max_chunks
        self.max_chars_per_chunk = max_chars_per_chunk
        self.max_recent_turns = max_recent_turns

        self.raw_text: str = ""
        self.chunks: List[str] = []
        self._chunk_tf: List[Dict[str, int]] = []
        self._df: Dict[str, int] = {}
        self._N: int = 0

        # Lightweight game state
        self.current_beat_index: int = 0
        self.flags: Dict[str, bool] = {}
        self.last_turn_ts: float = time.time()
        self.recent_turns: List[Tuple[str, str]] = []

        # Beats tuned for "A Distressed Damsel" (micro-adventure pacing)
        self.beats: List[StoryBeat] = [
            StoryBeat(
                "intro",
                "A desperate woman appears on the road",
                "A distraught woman, Layla, rushes from the woods pleading for help finding her missing son. She claims slavers attacked her family and a beast scattered the camp.",
                ["layla", "son", "slavers", "woods", "track", "help"],
                details=[
                    "Layla's clothes are torn and muddy, her breath shallow.",
                    "Fresh branches snap in the woods where she emerged.",
                    "Her eyes dart to the treeline as if expecting pursuit.",
                ],
                risk_low=[
                    "The road feels quiet; no immediate threat presses in.",
                ],
                risk_mid=[
                    "A faint rustle suggests someone or something may be nearby.",
                ],
                risk_high=[
                    "A sharp crack from the treeline hints danger could be close.",
                ],
            ),
            StoryBeat(
                "camp",
                "The abandoned camp in a clearing",
                "The party can track to a clearing with an abandoned camp-broken crates and a wagon-suggesting something is off. A trail leads deeper into the woods.",
                ["camp", "clearing", "wagon", "crates", "trail", "survival"],
                details=[
                    "The wagon sits skewed, one wheel splintered.",
                    "Crates are broken open, their contents gone.",
                    "Ashes in a cold firepit crumble at a touch.",
                ],
                risk_low=[
                    "The clearing is still, with no sign of movement.",
                ],
                risk_mid=[
                    "A faint, uneven track suggests haste and struggle.",
                ],
                risk_high=[
                    "The trail grows sharper and more frantic deeper into the woods.",
                ],
            ),
            StoryBeat(
                "cave_approach",
                "A cave mouth and a child's shoe",
                "Following the trail leads to a cave. Layla grows increasingly nervous. A child's shoe near the entrance raises the stakes.",
                ["cave", "shoe", "entrance", "nervous", "search"],
                details=[
                    "A small shoe lies half-buried in damp leaves.",
                    "Cool air seeps from the cave mouth.",
                    "Layla clutches her cloak tighter as you approach.",
                ],
                risk_low=[
                    "The cave entrance is silent, no movement inside.",
                ],
                risk_mid=[
                    "A faint drip echoes from within, obscuring other sounds.",
                ],
                risk_high=[
                    "Shallow claw marks scar the stone near the entrance.",
                ],
            ),
            StoryBeat(
                "cave_trap",
                "Inside the cave: trap and split passage",
                "Inside, the cave narrows. There is a spiked pit trap deeper in. The passage ends in a T-intersection: left or right.",
                ["trap", "pit", "spikes", "t", "left", "right", "listen"],
                details=[
                    "The tunnel walls tighten, forcing single file.",
                    "A sudden draft brushes past, carrying a stale scent.",
                    "The floor changes texture where the pit lies hidden.",
                ],
                risk_low=[
                    "The air is still, and the path seems safe for now.",
                ],
                risk_mid=[
                    "Loose gravel hints the ground may not be stable.",
                ],
                risk_high=[
                    "A misstep here could trigger the pit trap.",
                ],
            ),
            StoryBeat(
                "reveal",
                "The 'boy' and the werewolf ambush",
                "A boy sits calmly at a table-then the trap springs: werewolves ambush from behind and Layla reveals her true nature.",
                ["boy", "mother", "hungry", "werewolf", "ambush", "hybrid"],
                details=[
                    "The boy's gaze is steady, almost too calm.",
                    "A heavy silence hangs before the sudden attack.",
                    "Layla's posture shifts, predatory and cold.",
                ],
                risk_low=[
                    "You sense tension building but no immediate strike yet.",
                ],
                risk_mid=[
                    "A low growl rolls from the shadows behind you.",
                ],
                risk_high=[
                    "The ambush snaps shut with claws and teeth from the dark.",
                ],
            ),
            StoryBeat(
                "aftermath",
                "Aftermath and clues",
                "After the conflict, the party finds treasure and a ledger hinting at a broader criminal network. The ledger can connect to ongoing plots.",
                ["treasure", "ledger", "trade", "clues", "plot"],
                details=[
                    "A battered chest sits half-buried under hides.",
                    "The ledger is stained, pages stuck at the corners.",
                    "Coins glint faintly in the low light.",
                ],
                risk_low=[
                    "The cave settles into a wary quiet.",
                ],
                risk_mid=[
                    "Distant echoes suggest others could return.",
                ],
                risk_high=[
                    "Fresh tracks imply the threat may not be over.",
                ],
            ),
        ]

        self._load_script()
        self._build_index()

    # ------------------- loading / indexing -------------------

    def _load_script(self) -> None:
        if not self.script_path.exists():
            raise FileNotFoundError(f"Story script not found: {self.script_path}")

        if self.script_path.suffix.lower() == ".pdf":
            text = _try_extract_text_from_pdf(self.script_path)
        else:
            text = _normalize_ws(self.script_path.read_text(encoding="utf-8", errors="ignore"))

        # Remove the Open Game License block (not useful for gameplay prompting)
        text = re.split(r"\bOPEN GAME LICENSE\b", text, flags=re.IGNORECASE)[0]
        self.raw_text = _normalize_ws(text)

        self.chunks = _split_into_chunks(self.raw_text, max_chars=self.max_chars_per_chunk)
        self._N = len(self.chunks)

    def _build_index(self) -> None:
        """Build a tiny TF-IDF-like index for keyword retrieval."""
        self._chunk_tf = []
        self._df = {}

        for chunk in self.chunks:
            tf: Dict[str, int] = {}
            for tok in _tokenize(chunk):
                tf[tok] = tf.get(tok, 0) + 1
            self._chunk_tf.append(tf)
            for tok in tf.keys():
                self._df[tok] = self._df.get(tok, 0) + 1

    # ------------------- beat progression -------------------

    def _maybe_advance_beat(self, user_text: str) -> None:
        """
        Heuristic beat progression.
        For demos, you can also say: "next scene" / "continue" to force progress.
        """
        t = (user_text or "").lower()

        if "next scene" in t or "continue" in t or "go on" in t:
            self.current_beat_index = min(self.current_beat_index + 1, len(self.beats) - 1)
            return

        cur = self.beats[self.current_beat_index].beat_id

        # A few natural-language triggers
        if cur == "intro":
            if any(k in t for k in ["track", "follow", "trail", "search", "survival", "camp", "clearing"]):
                self.current_beat_index = min(self.current_beat_index + 1, len(self.beats) - 1)
        elif cur == "camp":
            if any(k in t for k in ["cave", "follow", "trail", "deeper", "woods"]):
                self.current_beat_index = min(self.current_beat_index + 1, len(self.beats) - 1)
        elif cur == "cave_approach":
            if any(k in t for k in ["enter", "go in", "inside", "cave", "torch"]):
                self.current_beat_index = min(self.current_beat_index + 1, len(self.beats) - 1)
        elif cur == "cave_trap":
            if any(k in t for k in ["left", "right", "listen", "voice", "mother"]):
                self.current_beat_index = min(self.current_beat_index + 1, len(self.beats) - 1)
        elif cur == "reveal":
            if any(k in t for k in ["after", "search", "loot", "treasure", "ledger"]):
                self.current_beat_index = min(self.current_beat_index + 1, len(self.beats) - 1)

    # ------------------- retrieval -------------------

    def _score_chunk(self, query_toks: List[str], chunk_tf: Dict[str, int]) -> float:
        """
        Simple TF-IDF score:
            score = sum_tf * idf
        where idf = log((N+1)/(df+1)) + 1
        """
        score = 0.0
        for tok in query_toks:
            tf = chunk_tf.get(tok, 0)
            if tf <= 0:
                continue
            df = self._df.get(tok, 0)
            idf = math.log((self._N + 1.0) / (df + 1.0)) + 1.0
            score += tf * idf
        return score

    def _retrieve(self, query: str, *, beat_bias: Optional[StoryBeat] = None, k: int = 3) -> List[str]:
        toks = _tokenize(query)
        if not toks or not self.chunks:
            return []

        # Add beat keywords to bias retrieval toward current scene
        if beat_bias:
            toks = toks + beat_bias.keywords

        scored: List[Tuple[float, int]] = []
        for i, tf in enumerate(self._chunk_tf):
            s = self._score_chunk(toks, tf)
            if s > 0:
                scored.append((s, i))

        scored.sort(reverse=True, key=lambda x: x[0])
        top = [self.chunks[i] for _, i in scored[:k]]
        return top

    # ------------------- public API -------------------

    def _pick_from_pool(self, pool: List[str], count: int) -> List[str]:
        if not pool or count <= 0:
            return []
        start = 0
        if pool:
            start = len(self.recent_turns) % len(pool)
        picked = []
        for i in range(min(count, len(pool))):
            picked.append(pool[(start + i) % len(pool)])
        return picked

    def _details_for_density(self, beat: StoryBeat, density: str) -> List[str]:
        if density == "high":
            count = 3
        elif density == "medium":
            count = 2
        else:
            count = 1
        return self._pick_from_pool(beat.details, count)

    def _risk_for_level(self, beat: StoryBeat, risk: str) -> List[str]:
        if risk == "high":
            pool = beat.risk_high
        elif risk == "normal":
            pool = beat.risk_mid
        else:
            pool = beat.risk_low
        return self._pick_from_pool(pool, 1)

    def get_context(self, user_text: str, *, extra_query: str = "", dm_controls: Optional[Dict[str, str]] = None) -> StoryContext:
        """
        Returns a small story context bundle to inject into the LLM prompt.
        """
        # Advance beat first (so retrieval follows the new beat)
        self._maybe_advance_beat(user_text)

        beat = self.beats[self.current_beat_index]
        query = f"{beat.title}. {beat.summary}. {user_text}. {extra_query}".strip()

        retrieved = self._retrieve(query, beat_bias=beat, k=self.max_chunks)

        controls = dict(dm_controls or {})
        density = controls.get("density", "medium")
        risk = controls.get("risk", "normal")
        beat_details = self._details_for_density(beat, density)
        risk_cues = self._risk_for_level(beat, risk)

        return StoryContext(
            beat_id=beat.beat_id,
            beat_title=beat.title,
            beat_summary=beat.summary,
            retrieved_passages=retrieved,
            flags=dict(self.flags),
            recent_turns=list(self.recent_turns),
            beat_details=beat_details,
            risk_cues=risk_cues,
            dm_controls=controls,
        )

    @staticmethod
    def _sanitize_turn_text(text: str, limit: int = 240) -> str:
        t = _normalize_ws(text or "")
        if len(t) <= limit:
            return t
        return t[: limit - 3].rstrip() + "..."

    def record_turn(self, user_text: str, dm_text: str) -> None:
        """
        Store recent dialogue turns to provide short-term memory to the LLM.
        """
        user_clean = self._sanitize_turn_text(user_text)
        dm_clean = self._sanitize_turn_text(dm_text)
        if not user_clean and not dm_clean:
            return
        self.recent_turns.append((user_clean, dm_clean))
        if len(self.recent_turns) > self.max_recent_turns:
            self.recent_turns = self.recent_turns[-self.max_recent_turns :]

    def set_flag(self, key: str, value: bool = True) -> None:
        self.flags[key] = bool(value)

    def reset(self) -> None:
        self.current_beat_index = 0
        self.flags = {}
        self.last_turn_ts = time.time()
        self.recent_turns = []
