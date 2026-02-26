# Offline Tarteel

Offline Quran verse recognition. Record someone reciting, identify the surah and ayah -- no internet required.

## Goal

Ship a model that runs on-device (phone or laptop) with **95%+ recall**, **sub-second latency**, and **under 200 MB** on disk. The current best approach (CTC forced alignment) hits 81% recall but weighs 1.2 GB and takes ~5s -- too large and too slow for a real product. Everything in this repo exists to close that gap.

## Design constraints

- **Offline-first.** No network calls at inference time. The model, index, and reference data all ship with the app.
- **Small models only.** Target < 200 MB total (model + any index). Phone storage is limited and download size matters.
- **Fast inference.** Under 1 second on Apple Silicon (MPS) or recent phone SoC. Users expect near-instant feedback after reciting.
- **Speaker-invariant.** Must work across accents, recording quality, and recitation styles -- not just professional studio audio from a single reciter.
- **Full Quran coverage.** All 6,236 verses, including short verses (3-4 words) that every approach currently struggles with.

## Current results

Benchmarked on 54 samples (user recordings, professional reference audio, crowdsourced recordings):

| Experiment | Approach | SeqAcc | Recall | Precision | Latency | Size |
|---|---|---|---|---|---|---|
| **ctc-alignment** | wav2vec2 CTC forced alignment (fine-tuned) | **81%** | **83%** | **83%** | ~5s | 1.2 GB |
| tarteel-whisper-base | tarteel-ai/whisper-base-ar-quran + QuranDB | 67% | 72% | 75% | ~3s | 290 MB |
| whisper-lora | Whisper-small + LoRA fine-tune + QuranDB | 58% | 64% | 65% | ~1.3s | 485 MB |

None of these hit the target. CTC alignment is the most accurate but 6x over the size budget and 5x over the latency budget. The ASR-based approaches are smaller but top out at 72% recall because Arabic transcription quality is the bottleneck.

## Next experiments

Three parallel experiments designed to break through the accuracy/size/speed tradeoff:

### Two-Stage Retrieval
Moonshine Tiny Arabic (27M params, 103 MB) does fast ASR to get a rough transcript, then a small CTC model re-scores only the top 50 verse candidates. This bounds the expensive CTC computation to 50 candidates instead of 6,236, getting CTC-level accuracy at a fraction of the latency.

### Distilled CTC
wav2vec2-base (95M params, ~380 MB, target ~95 MB after int8 quantization) fine-tuned for Arabic CTC, then knowledge-distilled from the large model. Same CTC scoring approach as the current best, but 3x smaller. The teacher-student distillation transfers the large model's discriminative power into the small model's architecture.

### Contrastive V2 (Audio Fingerprinting)
CLIP-style contrastive model mapping audio to a speaker-invariant 256-dim embedding. Pre-computed FAISS index of all 6,236 verses. At inference: one forward pass + nearest neighbor search = verse ID. No ASR, no text matching. Key improvements over v1: wav2vec2-base encoder (cross-lingual, not English-only HuBERT), deeper projection heads, multi-reciter training data, proper batch sizes (64-128 on GPU).

All three train on Modal A10G GPUs. Local Mac is inference-only.

## Project structure

```
shared/                  # Common utilities used by all experiments
  audio.py               # load_audio(path, sr=16000) -> float32 numpy array
  normalizer.py          # normalize_arabic(text) - strip diacritics, normalize alef/taa
  quran_db.py            # QuranDB - 6,236 verses, fuzzy match, multi-ayah spans

experiments/             # Each approach gets its own directory
  ctc-alignment/         # CTC forced alignment (current best, 81%)
  two-stage/             # Moonshine ASR + CTC re-score (in progress)
  distilled-ctc/         # wav2vec2-base knowledge-distilled (in progress)
  contrastive-v2/        # QuranCLAP v2 audio fingerprinting (in progress)
  whisper-lora/          # Whisper-small + LoRA adapter
  tarteel-whisper-base/  # Tarteel's whisper-base-ar-quran
  embedding-search/      # HuBERT + FAISS nearest-neighbor
  contrastive/           # QuranCLAP v1 (proof of concept)
  streaming-asr/         # mlx-whisper chunked streaming
  new-models/            # Multi-model benchmark (8 ASR models)

benchmark/               # Evaluation framework
  runner.py              # CLI benchmark runner
  test_corpus/           # 54 audio samples + manifest.json
  results/               # Timestamped JSON results + latest.json

data/                    # Reference data
  quran.json             # 6,236 verses (uthmani + cleaned text)
  reference_audio/       # EveryAyah samples (Alafasy)
  test_audio/            # User recordings

web/                     # Live demo
  server.py              # FastAPI backend
  frontend/              # React frontend

scripts/                 # Training scripts (Modal A10G GPU)
  train_ctc_base_modal.py    # wav2vec2-base CTC fine-tuning (shared by two-stage + distilled-ctc)
  train_distill_modal.py     # Knowledge distillation (large CTC -> small CTC)
  train_contrastive_v2_modal.py  # QuranCLAP v2 contrastive training
  train_modal.py         # LoRA training (whisper-lora experiment)
  train_lora.py          # Local LoRA training script (MPS/CUDA)

docs/plans/              # Design docs and experiment plans
REPORT.md                # Full experiment report with cross-comparison
RESEARCH-audio-to-verse.md  # Research notes on approaches
```

## All experiments

### ctc-alignment (current best -- 81% accuracy, 1.2 GB)

CTC forced alignment using a pre-trained Arabic wav2vec2 model. Scores candidate verses directly against frame-level character logits using the CTC forward algorithm, bypassing the information loss of greedy decoding.

**Flow:** audio -> wav2vec2 frame logits -> greedy decode -> Levenshtein top-100 candidates -> CTC re-score -> multi-verse span scoring -> best score wins

- **Model:** `jonatasgrosman/wav2vec2-large-xlsr-53-arabic` (1.2 GB)
- **Gap to target:** Accurate enough to prove the approach works, but too large (6x) and too slow (5x) for on-device use.

### two-stage (in progress)

Moonshine Tiny Arabic (27M) for fast ASR, then CTC forced-alignment re-scoring on just the top 50 candidates. Bounds the expensive CTC computation from 6,236 verses to 50.

- **Stage 1:** Moonshine Tiny Arabic (103 MB) -> transcript -> QuranDB.search(top_k=50)
- **Stage 2:** wav2vec2-base CTC (target ~95 MB after quantization) -> re-score 50 candidates
- **Target size:** ~200 MB combined
- **Requires:** Modal-trained wav2vec2-base CTC model (`scripts/train_ctc_base_modal.py`)

### distilled-ctc (in progress)

wav2vec2-base (95M params) with knowledge distillation from the large CTC model. Same scoring approach as ctc-alignment but 3x smaller.

- **Teacher:** wav2vec2-large-xlsr-53-arabic (315M params, 1.2 GB)
- **Student:** wav2vec2-base + Arabic CTC head (95M params, ~380 MB, target ~95 MB int8)
- **Loss:** alpha * CTC(student, ground_truth) + (1-alpha) * KL(student || teacher)
- **Requires:** Modal training (`scripts/train_ctc_base_modal.py`, then `scripts/train_distill_modal.py`)

### contrastive-v2 (in progress)

CLIP-style contrastive model (QuranCLAP v2). Maps audio to a speaker-invariant 256-dim embedding, matched against a pre-computed FAISS index of all 6,236 verses. One forward pass + nearest neighbor = verse ID. No ASR needed.

- **Audio encoder:** wav2vec2-base (cross-lingual, not English-only HuBERT like v1)
- **Text encoder:** AraBERT v02
- **Training:** Multi-reciter EveryAyah data, batch size 64-128, two-phase (frozen -> unfreeze last 2 layers)
- **Target size:** ~367 MB (audio encoder + projection + FAISS index)
- **Requires:** Modal training (`scripts/train_contrastive_v2_modal.py`)

### tarteel-whisper-base (67% accuracy, 290 MB)

Tarteel's Whisper-base fine-tuned for Quranic Arabic. Transcribe-then-match pipeline.

### whisper-lora (58% accuracy, 485 MB)

Whisper-small + LoRA adapter fine-tuned on EveryAyah + RetaSy. Transcribe-then-match pipeline.

### embedding-search (not viable standalone)

HuBERT embeddings + FAISS nearest neighbor. 100% same-reciter, 0% cross-speaker. HuBERT encodes speaker identity more than linguistic content.

### contrastive v1 (proof of concept, ~1.6% accuracy)

First attempt at CLIP-style audio-text matching. Failed due to English HuBERT on Arabic audio, batch size 4, single reciter. Contrastive-v2 addresses all of these.

### streaming-asr (43% accuracy, 6x faster to first match)

mlx-whisper chunked streaming. Streaming is good for UX (0.19s to first match) but accuracy is limited by whisper-base quality.

### new-models (model comparison)

Head-to-head benchmark of 8 ASR models. Key finding: Moonshine Tiny Arabic (103 MB) matches Whisper Large-v3-Turbo (3.1 GB) at 30x smaller. This is why the two-stage experiment uses Moonshine as Stage 1.

## Test corpus

`benchmark/test_corpus/manifest.json` contains 54 samples across three sources:

| Source | Count | Description |
|---|---|---|
| User recordings | 2 | Phone recordings (.m4a), ambient noise, non-professional |
| EveryAyah (Alafasy) | 23 | Professional studio recordings, includes long single-ayah and multi-ayah concatenated |
| RetaSy crowdsourced | 29 | Curated subset from 1,287 speakers across 81 countries |

**Categories:** short (17), medium (19), long (9), multi (9).

## Running benchmarks

```bash
# Activate the project venv
source .venv/bin/activate

# Run all registered experiments
.venv/bin/python -m benchmark.runner

# Run a single experiment
.venv/bin/python -m benchmark.runner --experiment ctc-alignment

# Filter by audio category
.venv/bin/python -m benchmark.runner --category short
```

Results are saved to `benchmark/results/<timestamp>.json`. The runner also maintains `benchmark/results/latest.json` with the best result per experiment.

## Adding a new experiment

1. Create `experiments/<name>/run.py` with two required functions:

```python
def predict(audio_path: str) -> dict:
    """Returns {"surah": int, "ayah": int, "ayah_end": int|None, "score": float, "transcript": str}"""

def model_size() -> int:
    """Total model size in bytes."""
```

2. Register it in `benchmark/runner.py`:

```python
EXPERIMENT_REGISTRY = {
    ...
    "your-experiment": EXPERIMENTS_DIR / "your-experiment" / "run.py",
}
```

3. Run the benchmark: `.venv/bin/python -m benchmark.runner --experiment your-experiment`

For multi-model experiments, also export `list_models()` and accept `model_name` as an optional parameter in `predict()` and `model_size()`.

## Shared utilities

All experiments use `from shared.X import ...`:

- `shared.audio.load_audio(path, sr=16000)` -- returns float32 numpy array at 16kHz
- `shared.normalizer.normalize_arabic(text)` -- strips diacritics, normalizes alef/taa marbuta/alef maqsura
- `shared.quran_db.QuranDB` -- loads `data/quran.json`, provides `match_verse(text)`, `search(text, top_k)`, `get_verse(surah, ayah)`

## Web demo

FastAPI backend + React frontend for live recitation and verse identification.

```bash
# Backend
.venv/bin/python web/server.py

# Frontend (separate terminal)
cd web/frontend && npm run dev
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Some experiments have additional dependencies (faiss-cpu, moonshine, mlx-whisper). Check individual experiment READMEs.

## Key findings

1. **CTC forced alignment is the most accurate approach** -- scoring candidates directly against frame logits avoids the information loss of greedy decoding, giving 81% accuracy. But the model is too large (1.2 GB) for on-device deployment.

2. **ASR quality is the bottleneck for transcribe-then-match approaches.** All ASR-based approaches fail on the same samples. The two-stage experiment sidesteps this by using ASR only for candidate retrieval, then CTC for the final decision.

3. **Embedding search needs speaker invariance.** HuBERT embeddings encode speaker identity more than linguistic content (100% same-reciter, 0% cross-speaker). Contrastive-v2 addresses this with multi-reciter contrastive training.

4. **Small models can match large ones.** Moonshine Tiny Arabic (103 MB) matches Whisper Large-v3-Turbo (3.1 GB) on our benchmark. Knowledge distillation should transfer the large CTC model's accuracy into a base-sized model.

5. **Short verses are hard across all approaches.** Verses under 3-4 words don't provide enough signal. May need minimum-length gating or surah-context bias in the final product.

## What we tried (and didn't work)

### Moonshine Tiny AR LoRA fine-tuning

Attempted to LoRA fine-tune [UsefulSensors/moonshine-tiny-ar](https://huggingface.co/UsefulSensors/moonshine-tiny-ar) (27M params, 103 MB) on Quran audio to improve its transcription for verse matching.

**What we tried:**
- LoRA r=8 on all attention (q/k/v/o_proj), lr=5e-4, 3000 steps
- LoRA r=4 encoder-only (decoder frozen), lr=1e-4, 2000 steps
- LoRA r=8 decoder-only, various learning rates
- Full fine-tuning (all 27M params), lr=2e-5, 3000 steps
- Diacritics stripped + EOS token appended (fixing data distribution mismatch)

Every configuration degraded the model. Base Moonshine scores 56% SeqAcc; the best fine-tuned variant dropped to 35-38%.

**Root cause:** Moonshine uses a LLaMA-derived tokenizer with only ~54 Arabic character tokens out of 32K vocab. Every Arabic character is its own token, so the decoder does character-by-character generation. Any weight perturbation (even LoRA r=4 at low LR) corrupts the character sequences, producing garbled output like `ب س الهر حا الحي` instead of `بسم الله الرحمن الرحيم`.

**Conclusion:** The base model (56% SeqAcc, 103 MB) is useful as-is for the two-stage pipeline's Stage 1 (rough transcript for candidate retrieval), but shouldn't be fine-tuned.

## Further reading

- `REPORT.md` -- Full experiment report with per-sample breakdowns, failure analysis, and recommendations
- `RESEARCH-audio-to-verse.md` -- Research survey of approaches (WavLink, Moonshine v2, WhisperKit, contrastive learning, audio fingerprinting)
- `docs/plans/` -- Design documents for individual experiments
- Individual experiment `README.md` files for reproduction instructions
