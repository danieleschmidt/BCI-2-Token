"""Tests for TokenDecoder."""

import numpy as np
import pytest

from bci2token.token_decoder import TokenDecoder, DEMO_VOCAB, DEMO_VOCAB_SIZE


class TestTokenDecoder:
    def setup_method(self):
        self.dec = TokenDecoder()

    def test_top_k_returns_k_tokens(self):
        logits = np.random.randn(DEMO_VOCAB_SIZE)
        result = self.dec.top_k(logits, k=5)
        assert len(result.tokens) == 5
        assert len(result.probabilities) == 5
        assert len(result.indices) == 5

    def test_probabilities_sum_to_one(self):
        """Top-k probs should sum to ≤ 1 (they're a subset of the full distribution)."""
        logits = np.random.randn(DEMO_VOCAB_SIZE)
        result = self.dec.top_k(logits, k=10)
        # Each prob in [0,1] and sum of top-k <= 1
        assert all(0 <= p <= 1 for p in result.probabilities)
        assert sum(result.probabilities) <= 1.0 + 1e-6

    def test_probabilities_descending_order(self):
        logits = np.random.randn(DEMO_VOCAB_SIZE)
        result = self.dec.top_k(logits, k=5)
        probs = result.probabilities
        assert probs == sorted(probs, reverse=True), "Probabilities should be in descending order"

    def test_top_k_tokens_in_vocab(self):
        logits = np.random.randn(DEMO_VOCAB_SIZE)
        result = self.dec.top_k(logits, k=5)
        for token in result.tokens:
            assert token in DEMO_VOCAB

    def test_high_confidence_logit_dominates(self):
        """Setting one logit very high should make it the top token."""
        logits = np.zeros(DEMO_VOCAB_SIZE)
        target_idx = 7
        logits[target_idx] = 100.0
        result = self.dec.top_k(logits, k=1)
        assert result.indices[0] == target_idx

    def test_temperature_effect(self):
        """Higher temperature should produce a more uniform distribution."""
        logits = np.random.randn(DEMO_VOCAB_SIZE)
        low_temp = TokenDecoder(temperature=0.1)
        high_temp = TokenDecoder(temperature=10.0)

        r_low = low_temp.top_k(logits, k=5)
        r_high = high_temp.top_k(logits, k=5)

        # Low temp: top prob should be higher (sharper)
        assert r_low.probabilities[0] > r_high.probabilities[0]

    def test_beam_search_returns_beams(self):
        frames = [np.random.randn(DEMO_VOCAB_SIZE) for _ in range(3)]
        beams = self.dec.beam_search(frames, beam_width=4, k=5)
        assert len(beams) == 4

    def test_beam_search_sequence_length(self):
        frames = [np.random.randn(DEMO_VOCAB_SIZE) for _ in range(3)]
        beams = self.dec.beam_search(frames, beam_width=4, k=5)
        for beam in beams:
            assert len(beam.token_ids) == 3  # one token per frame

    def test_beam_search_empty_frames(self):
        beams = self.dec.beam_search([], beam_width=4)
        assert beams == []

    def test_decode_sequence_readable(self):
        frames = [np.random.randn(DEMO_VOCAB_SIZE) for _ in range(2)]
        beams = self.dec.beam_search(frames, beam_width=2)
        sequences = self.dec.decode_sequence(beams)
        assert len(sequences) == 2
        for text, score in sequences:
            assert isinstance(text, str)
            assert isinstance(score, float)
            # Each decoded sequence should have 2 tokens (space-separated)
            assert len(text.split()) == 2
