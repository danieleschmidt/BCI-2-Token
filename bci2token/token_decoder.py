"""
TokenDecoder: Token logits → top-k tokens + beam search over short sequences

Supports:
  - Greedy top-k decoding with softmax probabilities
  - Beam search over multi-step sequences (stateless: given list of logit frames)
  - Optional vocabulary (defaults to a compact demo vocab)
"""

import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict


# ---- Vocabulary -------------------------------------------------------
# Compact 512-token vocabulary for demo use.
# For production: swap in tiktoken or transformers tokenizer.

def _build_demo_vocab() -> List[str]:
    """Build a small, readable demo vocabulary."""
    tokens = []
    # Common English words likely to surface in mental-imagery tasks
    words = [
        # Mental state / cognitive words
        "think", "mind", "focus", "relax", "move", "stop", "yes", "no",
        "left", "right", "up", "down", "start", "end", "go", "wait",
        # Nouns
        "hand", "arm", "foot", "eye", "brain", "signal", "wave", "data",
        "image", "word", "sound", "light", "color", "shape", "object",
        # Verbs
        "see", "hear", "feel", "touch", "imagine", "remember", "forget",
        "open", "close", "push", "pull", "reach", "grasp", "release",
        # Functional tokens
        "<unk>", "<bos>", "<eos>", "<pad>",
        # Alphabet / syllables
        *[chr(c) for c in range(ord('a'), ord('z') + 1)],
        # Digits
        *[str(d) for d in range(10)],
        # Fill to 512 with indexed tokens
        *[f"tok_{i}" for i in range(512 - 70)],
    ]
    return words[:512]


DEMO_VOCAB: List[str] = _build_demo_vocab()
DEMO_VOCAB_SIZE = len(DEMO_VOCAB)


@dataclass
class DecodedResult:
    """Result of top-k decoding for a single EEG frame."""
    tokens: List[str]            # top-k token strings
    probabilities: List[float]   # corresponding probabilities
    indices: List[int]           # vocabulary indices

    def __repr__(self):
        pairs = [(t, f"{p:.4f}") for t, p in zip(self.tokens, self.probabilities)]
        return f"DecodedResult({pairs})"


@dataclass
class BeamHypothesis:
    """A single beam search hypothesis."""
    token_ids: List[int] = field(default_factory=list)
    score: float = 0.0           # cumulative log-prob

    def extend(self, token_id: int, log_prob: float) -> "BeamHypothesis":
        return BeamHypothesis(
            token_ids=self.token_ids + [token_id],
            score=self.score + log_prob,
        )

    @property
    def normalized_score(self) -> float:
        length = max(len(self.token_ids), 1)
        return self.score / length


class TokenDecoder:
    """
    Converts EEG-derived token logits into human-readable token predictions.

    Args:
        vocab:      List of token strings. Defaults to DEMO_VOCAB.
        vocab_size: If vocab is not provided, truncate DEMO_VOCAB to this size.
        temperature: Softmax temperature (higher = more uniform, lower = sharper).
    """

    def __init__(
        self,
        vocab: Optional[List[str]] = None,
        vocab_size: Optional[int] = None,
        temperature: float = 1.0,
    ):
        if vocab is not None:
            self.vocab = vocab
        elif vocab_size is not None:
            self.vocab = DEMO_VOCAB[:vocab_size]
        else:
            self.vocab = DEMO_VOCAB
        self.vocab_size = len(self.vocab)
        self.temperature = temperature

    def top_k(self, logits: np.ndarray, k: int = 5) -> DecodedResult:
        """
        Decode top-k tokens from a logit vector.

        Args:
            logits: shape (vocab_size,)
            k:      number of top tokens to return

        Returns:
            DecodedResult with tokens, probabilities, indices
        """
        if len(logits) != self.vocab_size:
            # Slice or pad to match vocab_size
            logits = logits[: self.vocab_size]

        t = torch.tensor(logits, dtype=torch.float32)
        probs = F.softmax(t / self.temperature, dim=0).numpy()

        top_idx = np.argsort(probs)[::-1][:k]
        return DecodedResult(
            tokens=[self.vocab[i] for i in top_idx],
            probabilities=[float(probs[i]) for i in top_idx],
            indices=[int(i) for i in top_idx],
        )

    def beam_search(
        self,
        logit_frames: List[np.ndarray],
        beam_width: int = 4,
        k: int = 5,
    ) -> List[BeamHypothesis]:
        """
        Beam search over a sequence of logit frames (one per EEG window).

        Each frame is treated as a step in an auto-regressive decode; we greedily
        expand the best `beam_width` hypotheses at each step using the per-frame
        logit distributions.  This is a *non-neural* beam search: it doesn't use
        an LM transition model — each step is independent, which is appropriate
        for the current open-loop BCI setting.

        Args:
            logit_frames: list of arrays, each shape (vocab_size,) — one per window
            beam_width:   number of beams to maintain
            k:            top-k candidates to consider at each step

        Returns:
            Sorted list of BeamHypothesis (best first by normalized log-prob)
        """
        if not logit_frames:
            return []

        # Initialize beams from first frame
        first_probs = F.softmax(
            torch.tensor(logit_frames[0][: self.vocab_size], dtype=torch.float32)
            / self.temperature,
            dim=0,
        ).numpy()
        top_ids = np.argsort(first_probs)[::-1][:beam_width]
        beams = [
            BeamHypothesis(
                token_ids=[int(i)],
                score=float(np.log(first_probs[i] + 1e-10)),
            )
            for i in top_ids
        ]

        for frame in logit_frames[1:]:
            probs = F.softmax(
                torch.tensor(frame[: self.vocab_size], dtype=torch.float32)
                / self.temperature,
                dim=0,
            ).numpy()
            candidates: List[BeamHypothesis] = []
            top_ids = np.argsort(probs)[::-1][:k]
            for beam in beams:
                for tid in top_ids:
                    candidates.append(
                        beam.extend(int(tid), float(np.log(probs[tid] + 1e-10)))
                    )
            # Keep top beam_width by normalized score
            candidates.sort(key=lambda h: h.normalized_score, reverse=True)
            beams = candidates[:beam_width]

        return beams

    def decode_sequence(
        self, beams: List[BeamHypothesis]
    ) -> List[Tuple[str, float]]:
        """
        Convert beam hypotheses to readable (token_string, score) pairs.
        """
        return [
            (
                " ".join(self.vocab[i] for i in h.token_ids),
                h.normalized_score,
            )
            for h in beams
        ]
