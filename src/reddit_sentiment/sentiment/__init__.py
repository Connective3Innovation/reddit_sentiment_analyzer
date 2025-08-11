"""Engine‑agnostic sentiment facade with *true* lazy loading.

Nothing heavy (HuggingFace, tiktoken, protobuf, etc.) is imported at module
import‑time.  Only when you first request `engine="hf"` do we try to pull those
libs.  That keeps test discovery fast and avoids multi‑hundred‑MB downloads.
"""
from __future__ import annotations

from typing import Literal, Sequence

import pandas as pd

# Cache of already‑instantiated engines
_ENGINE_CACHE: dict[str, "BaseEngine"] = {}


class _UnavailableEngine(Exception):
    """Raised when the user asks for an engine whose deps are missing."""


# ──────────────────────────────────────────────────────────────────────────────
# Lazy loader helper
# ──────────────────────────────────────────────────────────────────────────────

def _get_engine(name: str):
    if name in _ENGINE_CACHE:  # already built
        return _ENGINE_CACHE[name]

    if name == "vader":
        from .vader_engine import VaderEngine  # local import is light

        _ENGINE_CACHE[name] = VaderEngine()
        return _ENGINE_CACHE[name]

    if name == "hf":
        try:
            from .hf_engine import HfEngine  # heavy! import only when asked
        except Exception as err:  # pragma: no cover
            raise _UnavailableEngine(
                "HuggingFace engine requested but optional dependencies are missing.\n"
                
                "Install deps: transformers, torch, protobuf, sentencepiece, tiktoken."
            ) from err

        _ENGINE_CACHE[name] = HfEngine()
        return _ENGINE_CACHE[name]

    raise KeyError(f"Unknown engine: {name}")


EngineName = Literal["vader", "hf"]


def analyze(texts: Sequence[str], engine: EngineName = "vader") -> pd.DataFrame:  # noqa: WPS110
    """Return sentiment scores as a DataFrame for *texts* using chosen engine."""
    eng = _get_engine(engine)
    return eng.run(texts)