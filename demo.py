#!/usr/bin/env python3
"""
BCI-2-Token Demo
================
Demonstrates the full EEG → token logit → top-5 token decoding pipeline
for 5 different mental states, with differential privacy applied.

Run: python demo.py
"""

import numpy as np
import sys

from bci2token import (
    EEGTokenizer,
    TokenDecoder,
    PrivacyFilter,
    SyntheticEEGGenerator,
)
from bci2token.synthetic_eeg import MENTAL_STATES


def separator(char="─", width=60):
    print(char * width)


def main():
    print("\n🧠 BCI-2-Token: EEG → Language Model Token Probabilities")
    separator("═")

    # ── Configuration ──────────────────────────────────────────────
    N_CHANNELS = 32
    FS = 256.0        # Hz
    DURATION = 2.0    # seconds
    VOCAB_SIZE = 512  # demo vocab
    TOP_K = 5

    EPSILON = 1.0     # DP privacy budget
    DELTA = 1e-5      # DP failure probability

    # ── Initialize pipeline components ─────────────────────────────
    print("\n⚙  Initializing pipeline...")
    gen = SyntheticEEGGenerator(n_channels=N_CHANNELS, fs=FS, duration_s=DURATION)
    tok = EEGTokenizer(n_channels=N_CHANNELS, vocab_size=VOCAB_SIZE, fs=FS)
    dec = TokenDecoder(vocab_size=VOCAB_SIZE)  # uses DEMO_VOCAB
    pf  = PrivacyFilter(epsilon=EPSILON, delta=DELTA, sensitivity=1.0)

    print(f"   EEGTokenizer  : {N_CHANNELS} channels × {int(FS*DURATION)} samples → {VOCAB_SIZE} token logits")
    print(f"   TokenDecoder  : vocab size {dec.vocab_size}, temperature {dec.temperature}")
    print(f"   PrivacyFilter : ε={EPSILON}, δ={DELTA}, noise σ={pf.sigma:.4f}")

    # ── Decode each mental state ────────────────────────────────────
    print(f"\n{'Mental State':<28}  Top-{TOP_K} Predicted Tokens + Probabilities")
    separator()

    results = {}
    for state in MENTAL_STATES:
        trial = gen.generate(state)
        logits = tok.encode(trial.data)              # raw logits (vocab_size,)
        private_logits = pf.apply(logits)            # DP-protected logits
        result = dec.top_k(private_logits, k=TOP_K)  # top-k decode
        results[state] = result

        token_str = "  ".join(
            f"{t}({p:.3f})" for t, p in zip(result.tokens, result.probabilities)
        )
        print(f"  {state:<26}  {token_str}")

    separator()

    # ── Per-state probability distributions ─────────────────────────
    print(f"\n📊 Per-State Top-{TOP_K} Token Distribution\n")
    for state, result in results.items():
        print(f"  [{state}]")
        for token, prob, idx in zip(result.tokens, result.probabilities, result.indices):
            bar = "█" * int(prob * 200)
            print(f"    {token:<18}  {prob:.4f}  {bar}")
        print()

    # ── Beam Search Demo ────────────────────────────────────────────
    print("🔭 Beam Search Demo (3 EEG windows → token sequence)")
    separator()

    # Use mental_arithmetic — generates 3 overlapping windows
    windows = [gen.generate("mental_arithmetic").data for _ in range(3)]
    logit_frames = [tok.encode(w) for w in windows]
    beams = dec.beam_search(logit_frames, beam_width=4, k=8)
    sequences = dec.decode_sequence(beams)

    for rank, (seq, score) in enumerate(sequences, 1):
        print(f"  Beam {rank}: [{seq}]  score={score:.4f}")

    # ── Privacy Budget Summary ──────────────────────────────────────
    print(f"\n🔐 Privacy Budget Summary")
    separator()
    print(f"  {pf.budget}")
    print(f"  Queries made: {pf.budget._num_queries}")
    print(f"  Remaining ε:  {pf.budget.remaining_epsilon:.4f}")

    print("\n✅ Done.\n")


if __name__ == "__main__":
    main()
