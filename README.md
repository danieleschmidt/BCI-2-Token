# BCI-2-Token

**EEG brain signals → language model token probabilities**

A focused Python pipeline that converts raw EEG (electroencephalography) recordings directly into LLM-compatible token logit vectors — with differential privacy protection built in.

```
EEG (channels × time)
        ↓
  Spectral feature extraction
  (delta / theta / alpha / beta / gamma band powers)
        ↓
  Learned linear projection (PyTorch)
        ↓
  PrivacyFilter  →  (ε,δ)-DP Gaussian noise
        ↓
  Token logits  →  Top-k tokens + Beam search
```

---

## Why

Brain-computer interfaces generate rich neural signals but translating them into discrete language tokens is non-trivial. This library provides the missing link: a minimal, tested, privacy-aware pipeline that takes raw EEG and produces the probability distributions that language models expect.

Key use cases:
- Non-invasive BCI assistive communication (motor imagery → text)
- EEG-grounded language model conditioning
- Privacy-preserving neural data research

---

## Architecture

### `EEGTokenizer`
Converts a raw EEG trial (shape `channels × timesteps`) into a token logit vector:

1. **Band power extraction** — Welch's periodogram over 5 bands: delta (0.5–4 Hz), theta (4–8 Hz), alpha (8–13 Hz), beta (13–30 Hz), gamma (30–45 Hz)
2. **Log-normalization** — `log1p` for numerical stability
3. **Linear projection** — `MLP(spectral_features) → token_logits` via a 2-layer PyTorch network with LayerNorm + GELU

### `TokenDecoder`
Converts raw token logits into readable predictions:

- **Top-k decoding** — softmax + argsort, configurable temperature
- **Beam search** — stateless multi-window beam search over sequential EEG frames

### `PrivacyFilter`
Differential privacy via the Gaussian Mechanism (Dwork & Roth §A.1):

```
σ = Δf × √(2 ln(1.25/δ)) / ε
```

Supports L2 sensitivity clipping, budget tracking, and batch processing.

### `SyntheticEEGGenerator`
Generates realistic test EEG based on empirical spectral profiles:

| Mental State | Dominant Signature |
|---|---|
| `rest` | Strong alpha (8–13 Hz) |
| `motor_imagery_left` | Alpha/beta ERD, slight gamma |
| `motor_imagery_right` | Alpha/beta ERD (lateralized) |
| `mental_arithmetic` | Elevated frontal theta, suppressed alpha |
| `visual_attention` | Elevated occipital gamma |

---

## Quickstart

```python
import numpy as np
from bci2token import EEGTokenizer, TokenDecoder, PrivacyFilter, SyntheticEEGGenerator

# Generate synthetic EEG for a mental state
gen = SyntheticEEGGenerator(n_channels=32, fs=256.0)
trial = gen.generate("mental_arithmetic")  # shape (32, 512)

# Encode to token logits
tok = EEGTokenizer(n_channels=32, vocab_size=50257, fs=256.0)
logits = tok.encode(trial.data)  # shape (50257,)

# Apply differential privacy
pf = PrivacyFilter(epsilon=1.0, delta=1e-5)
private_logits = pf.apply(logits)

# Decode top-5 tokens
dec = TokenDecoder()
result = dec.top_k(private_logits, k=5)
print(result)
# DecodedResult([('think', '0.0823'), ('focus', '0.0791'), ...])
```

---

## Demo

```bash
python demo.py
```

Generates EEG for 5 mental states, runs the full pipeline with DP, and prints per-state token distributions + a beam search example.

---

## Tests

```bash
python -m pytest tests/ -v
```

45 tests covering spectral extraction, tokenizer output shapes, privacy math (σ formula, budget tracking), synthetic EEG spectral properties (rest → alpha dominant, arithmetic → theta dominant), and decoder correctness.

---

## Installation

```bash
pip install -e .
```

**Requirements:** Python ≥ 3.10, PyTorch ≥ 2.0, NumPy ≥ 1.24

---

## Extending

**Plug in a real tokenizer:**
```python
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
decoder = TokenDecoder(vocab=[enc.decode([i]) for i in range(enc.n_vocab)])
```

**Train the projection model:**
```python
# EEGTokenizer.model is a standard nn.Module
# Provide (EEG, target_token_id) pairs and train with CrossEntropyLoss
optimizer = torch.optim.AdamW(tok.model.parameters(), lr=1e-4)
```

**Use real EEG:**
```python
# Load MNE/EDF data
import mne
raw = mne.io.read_raw_edf("subject01.edf", preload=True)
eeg_array = raw.get_data()  # (channels, timesteps)
logits = tok.encode(eeg_array)
```

---

## Privacy

This library uses the **Gaussian Mechanism** for (ε, δ)-differential privacy. For medical/neuroscience deployments:

- Use `epsilon ≤ 1.0` for strong privacy
- Set `delta = 1/n²` where `n` is your dataset size
- Enable `clip_norm` to bound L2 sensitivity empirically
- Track budget across sessions with `PrivacyBudget`

---

## License

MIT
