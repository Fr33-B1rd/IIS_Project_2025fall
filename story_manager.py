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


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


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


@dataclass
class StoryContext:
    beat_id: str
    beat_title: str
    beat_summary: str
    retrieved_passages: List[str]
    flags: Dict[str, bool]


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
        use_time: bool = True,
    ):
        self.script_path = Path(script_path)
        self.max_chunks = max_chunks
        self.max_chars_per_chunk = max_chars_per_chunk
        self.use_time = use_time

        self.raw_text: str = ""
        self.chunks: List[str] = []
        self._chunk_tf: List[Dict[str, int]] = []
        self._df: Dict[str, int] = {}
        self._N: int = 0

        # Lightweight game state
        self.current_beat_index: int = 0
        self.flags: Dict[str, bool] = {}
        self.last_turn_ts: float = time.time()

        # Beats tuned for "A Distressed Damsel" (micro-adventure pacing)
        self.beats: List[StoryBeat] = [
            StoryBeat(
                "intro",
                "A desperate woman appears on the road",
                "A distraught woman, Layla, rushes from the woods pleading for help finding her missing son. She claims slavers attacked her family and a beast scattered the camp.",
                ["layla", "son", "slavers", "woods", "track", "help"],
            ),
            StoryBeat(
                "camp",
                "The abandoned camp in a clearing",
                "The party can track to a clearing with an abandoned camp—broken crates and a wagon—suggesting something is off. A trail leads deeper into the woods.",
                ["camp", "clearing", "wagon", "crates", "trail", "survival"],
            ),
            StoryBeat(
                "cave_approach",
                "A cave mouth and a child's shoe",
                "Following the trail leads to a cave. Layla grows increasingly nervous. A child's shoe near the entrance raises the stakes.",
                ["cave", "shoe", "entrance", "nervous", "search"],
            ),
            StoryBeat(
                "cave_trap",
                "Inside the cave: trap and split passage",
                "Inside, the cave narrows. There is a spiked pit trap deeper in. The passage ends in a T-intersection: left or right.",
                ["trap", "pit", "spikes", "t", "left", "right", "listen"],
            ),
            StoryBeat(
                "reveal",
                "The 'boy' and the werewolf ambush",
                "A boy sits calmly at a table—then the trap springs: werewolves ambush from behind and Layla reveals her true nature.",
                ["boy", "mother", "hungry", "werewolf", "ambush", "hybrid"],
            ),
            StoryBeat(
                "aftermath",
                "Aftermath and clues",
                "After the conflict, the party finds treasure and a ledger hinting at a broader criminal network. The ledger can connect to ongoing plots.",
                ["treasure", "ledger", "trade", "clues", "plot"],
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

    def get_context(self, user_text: str, *, extra_query: str = "") -> StoryContext:
        """
        Returns a small story context bundle to inject into the LLM prompt.
        """
        # Advance beat first (so retrieval follows the new beat)
        self._maybe_advance_beat(user_text)

        beat = self.beats[self.current_beat_index]
        query = f"{beat.title}. {beat.summary}. {user_text}. {extra_query}".strip()

        retrieved = self._retrieve(query, beat_bias=beat, k=self.max_chunks)

        return StoryContext(
            beat_id=beat.beat_id,
            beat_title=beat.title,
            beat_summary=beat.summary,
            retrieved_passages=retrieved,
            flags=dict(self.flags),
        )

    def set_flag(self, key: str, value: bool = True) -> None:
        self.flags[key] = bool(value)

    def reset(self) -> None:
        self.current_beat_index = 0
        self.flags = {}
        self.last_turn_ts = time.time()
