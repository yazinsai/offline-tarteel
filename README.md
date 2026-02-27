# Offline Tarteel

Offline Quran verse recognition. Record someone reciting, identify the surah and ayah -- no internet required.

## Goal

Ship a model that runs on-device (phone or laptop) with **95%+ recall**, **sub-second latency**, and **under 200 MB** on disk. The current best approach (`nvidia-fastconformer`) reaches **87% recall** at **115 MB** and **0.33s** latency, but still misses the 95% recall bar. Everything in this repo exists to close that final gap.

## Design constraints

- **Offline-first.** No network calls at inference time. The model, index, and reference data all ship with the app.
- **Small models only.** Target < 200 MB total (model + any index). Phone storage is limited and download size matters.
- **Fast inference.** Under 1 second on Apple Silicon (MPS) or recent phone SoC. Users expect near-instant feedback after reciting.
- **Speaker-invariant.** Must work across accents, recording quality, and recitation styles -- not just professional studio audio from a single reciter.
- **Full Quran coverage.** All 6,236 verses, including short verses (3-4 words) that every approach currently struggles with.

## Current results

Benchmarked on 54 samples (user recordings, professional reference audio, crowdsourced recordings):

As of **February 26, 2026**, the table below reflects best full-corpus (54-sample) runs.

| Experiment | Approach | SeqAcc | Recall | Precision | Latency | Size |
|---|---|---|---|---|---|---|
| **nvidia-fastconformer** | NeMo FastConformer transcript + span-aware QuranDB matching | **85%** | **87%** | **89%** | **0.33s** | **115 MB** |
| **ctc-alignment** | wav2vec2 CTC forced alignment (large baseline) | 81% | 83% | 83% | 3.24s | 1.2 GB |
| rabah-pruned-ctc/8-layer-ft-fn-int8 | Rabah Quran CTC, 8L first_n pruned + fine-tuned + int8 | **72%** | 74% | 73% | 7.02s | **145 MB** |
| rabah-pruned-ctc/12-layer-ft-es-int8 | Rabah Quran CTC, 12L evenly_spaced pruned + fine-tuned + int8 | **72%** | 73% | 74% | 3.65s | 193 MB |
| two-stage | Moonshine ASR + large CTC re-score | 72% | 73% | 74% | 9.48s | 1.3 GB |
| two-stage-faster-whisper-pruned | faster-whisper Quran + fine-tuned pruned CTC re-score | 70% | 73% | 73% | 10.06s | 582 MB |
| tarteel-whisper-base | tarteel-ai/whisper-base-ar-quran + QuranDB | 67% | 72% | 75% | ~3s | 290 MB |
| whisper-lora | Whisper-small + LoRA fine-tune + QuranDB | 58% | 64% | 65% | ~1.3s | 485 MB |
| rabah-pruned-ctc/8-layer-ft-es-int8 | 8L evenly_spaced (wrong strategy) | 56% | 57% | 57% | 6.97s | 145 MB |
| rabah-pruned-ctc/6-layer-ft-es-int8 | 6L evenly_spaced pruned + fine-tuned + int8 | 48% | 51% | 51% | 6.63s | 121 MB |

`nvidia-fastconformer` remains the best model overall. The **fine-tuned pruned Rabah models** are the major new result: `8-layer-ft-fn-int8` reaches **72% SeqAcc at 145 MB** (under the 200 MB target), up from 12% before fine-tuning. The `first_n` pruning strategy (keep layers 0-7) massively outperforms `evenly_spaced` (72% vs 56%), likely because contiguous early layers preserve better feature flow.

## Experiment status

- **NVIDIA FastConformer** is the top model: 85% SeqAcc, 115 MB, 0.33s latency.
- **Rabah pruned+fine-tuned path now works.** Fine-tuning the CTC head on pruned representations recovered accuracy from 12% to 72% (8-layer first_n). The 8L int8 model is 145 MB -- well under the 200 MB target. The key insight: `first_n` pruning (keep layers 0-7) vastly outperforms `evenly_spaced` (72% vs 56%).
- **Two-stage faster-whisper path** reaches 70% SeqAcc using the fine-tuned 8L CTC as Stage 2, but is still too large/slow (582 MB, 10s).

### Two-Stage Retrieval (historical 72% setup; current pruned variant at 70%)
Moonshine Tiny Arabic (27M params, 103 MB) does fast ASR to get a rough transcript, then CTC forced-alignment re-scores only the top 50 verse candidates. This bounds the expensive CTC computation to 50 candidates instead of 6,236. Currently falls back to the large CTC model (1.2 GB) because the small CTC model failed to train (see "What we tried"). With a working small CTC re-scorer (~95 MB), this would hit the size target.

### Distilled CTC (failed)
wav2vec2-base (95M params) knowledge-distilled from the large CTC model. The base model can't learn Arabic CTC -- English-only pretraining means no Arabic speech representations to build on (see "What we tried"). The smallest multilingual alternatives (MMS-300M, XLS-R-300M) are ~300M params, barely smaller than the 317M large model, which defeats the purpose.

### Contrastive V2 (failed)
CLIP-style contrastive model mapping audio to a speaker-invariant 256-dim embedding. Trained for 8 epochs (6 frozen + 2 unfrozen) on 30k EveryAyah samples on A100-80GB. Validation accuracy stuck at ~9% (random chance = 3.1% with batch 32) -- the model memorizes training data but doesn't generalize. Same root cause: wav2vec2-base doesn't produce useful Arabic audio features. See "What we tried" below.

## Project structure

```
shared/                  # Common utilities used by all experiments
  audio.py               # load_audio(path, sr=16000) -> float32 numpy array
  normalizer.py          # normalize_arabic(text) - strip diacritics, normalize alef/taa
  quran_db.py            # QuranDB - 6,236 verses, fuzzy match, multi-ayah spans

experiments/             # Each approach gets its own directory
  ctc-alignment/         # CTC forced alignment (strong baseline, 81%)
  two-stage/             # Moonshine ASR + CTC re-score (72%, large model fallback)
  two-stage-faster-whisper-pruned/  # faster-whisper Quran + pruned CTC re-score
  distilled-ctc/         # wav2vec2-base knowledge-distilled (failed)
  rabah-pruned-ctc/      # Rabah Quran CTC (12/8/6 + fine-tuned int8 variants)
  nvidia-fastconformer/  # NeMo FastConformer Arabic benchmark
  contrastive-v2/        # QuranCLAP v2 audio fingerprinting (failed)
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

scripts/                 # Training scripts (Modal A100-80GB GPU)
  train_pruned_ctc_modal.py    # Fine-tune pruned Rabah CTC models (the key training script)
  quantize_pruned_models.py    # PyTorch/ONNX int8 quantization
  build_rabah_pruned_models.py # Build naive-pruned Rabah checkpoints
  train_ctc_base_modal.py      # wav2vec2-base CTC fine-tuning (failed -- see "What we tried")
  train_distill_modal.py       # Knowledge distillation (blocked on CTC base)
  train_contrastive_v2_modal.py  # QuranCLAP v2 contrastive training
  train_modal.py               # LoRA training (whisper-lora experiment)
  train_lora.py                # Local LoRA training script (MPS/CUDA)

docs/plans/              # Design docs and experiment plans
REPORT.md                # Full experiment report with cross-comparison
RESEARCH-audio-to-verse.md  # Research notes on approaches
```

## All experiments

### ctc-alignment (strong baseline -- 81% accuracy, 1.2 GB)

CTC forced alignment using a pre-trained Arabic wav2vec2 model. Scores candidate verses directly against frame-level character logits using the CTC forward algorithm, bypassing the information loss of greedy decoding.

**Flow:** audio -> wav2vec2 frame logits -> greedy decode -> Levenshtein top-100 candidates -> CTC re-score -> multi-verse span scoring -> best score wins

- **Model:** `jonatasgrosman/wav2vec2-large-xlsr-53-arabic` (1.2 GB)
- **Gap to target:** Accurate enough to prove the approach works, but too large (6x) and too slow (5x) for on-device use.

### two-stage (72% SeqAcc -- large CTC fallback)

Moonshine Tiny Arabic (27M) for fast ASR, then CTC forced-alignment re-scoring on just the top 50 candidates. Bounds the expensive CTC computation from 6,236 verses to 50.

- **Stage 1:** Moonshine Tiny Arabic (103 MB) -> transcript -> QuranDB.search(top_k=50)
- **Stage 2:** CTC re-score 50 candidates (currently falls back to large model)
- **Result:** 72% SeqAcc, 78% recall, 78% precision with large CTC re-scorer
- **Blocker:** No working small CTC model. wav2vec2-base can't learn Arabic (see "What we tried").
- **Target size:** ~200 MB with a working small CTC model

### distilled-ctc (blocked)

wav2vec2-base (95M params) with knowledge distillation from the large CTC model. Same scoring approach as ctc-alignment but 3x smaller.

- **Teacher:** wav2vec2-large-xlsr-53-arabic (315M params, 1.2 GB)
- **Student:** wav2vec2-base + Arabic CTC head (95M params, ~380 MB, target ~95 MB int8)
- **Status:** Blocked. The student model (wav2vec2-base) can't learn Arabic CTC -- English-only pretraining means no Arabic speech representations to build on. See "What we tried" below.

### rabah-pruned-ctc (fine-tuned variants are the key result)

Rabah's Quran-specific wav2vec2-large checkpoint, layer-pruned and fine-tuned for small on-device CTC alignment.

- **Source model:** `rabah2026/wav2vec2-large-xlsr-53-arabic-quran-v_final` (24 layers, 1.2 GB)
- **Scoring:** Same CTC forced-alignment stack as `ctc-alignment`
- **Fine-tuned variants:** `8-layer-ft-fn-int8` (72%, 145 MB), `12-layer-ft-es-int8` (72%, 193 MB), `8-layer-ft-es-int8` (56%, 145 MB), `6-layer-ft-es-int8` (48%, 121 MB)
- **Naive-pruned baselines:** `12/8/6-layer-int8` (accuracy collapses without fine-tuning -- CTC head was trained on layer-24 representations)
- **Training:** `scripts/train_pruned_ctc_modal.py` on Modal A100-80GB. 5000 steps, LR 3e-5, EveryAyah + RetaSy, freeze CNN + lower half of transformer.
- **Key finding:** `first_n` pruning (keep layers 0-7) gets 72% vs 56% for `evenly_spaced` (keep layers 0,3,7,10,13,16,20,23). Contiguous early layers preserve better feature propagation.

### nvidia-fastconformer (new)

Arabic FastConformer hybrid model via NeMo:

- **Model:** `nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0`
- **Pipeline:** audio -> FastConformer transcript -> QuranDB span-aware match
- **Dependency:** `nemo_toolkit[asr]` (optional extra `.[nemo]`)

### two-stage-faster-whisper-pruned (new)

Two-stage retrieval variant using faster-whisper for ASR and fine-tuned pruned CTC for re-scoring:

- **Stage 1:** `OdyAsh/faster-whisper-base-ar-quran` (147 MB, CTranslate2 int8)
- **Stage 2:** CTC re-score top-50 candidates using fine-tuned 8L Rabah CTC (435 MB fp32)
- **Result:** 70% SeqAcc / 73% recall / 73% precision, 10s latency, 582 MB total
- **Bottleneck:** Stage 2 CTC model is still large; would benefit from int8 quantization or further pruning

### contrastive-v2 (failed -- val accuracy stuck at 9%)

CLIP-style contrastive model (QuranCLAP v2). Maps audio to a speaker-invariant 256-dim embedding, matched against a pre-computed FAISS index of all 6,236 verses. One forward pass + nearest neighbor = verse ID. No ASR needed.

- **Audio encoder:** wav2vec2-base (95M params)
- **Text encoder:** AraBERT v02 (136M params)
- **Training:** 30k samples from EveryAyah, batch 32 (effective 128 via grad accum), two-phase (frozen -> unfreeze last 2 layers), A100-80GB
- **Result:** Val accuracy stuck at ~9% after 8 epochs. Model overfits (train acc 14%, val acc 9%). wav2vec2-base can't produce useful Arabic audio representations.
- **Target size:** ~367 MB (audio encoder + projection + FAISS index)

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

# Run Rabah pruned variants (expanded via list_models)
.venv/bin/python -m benchmark.runner --experiment rabah-pruned-ctc
.venv/bin/python -m benchmark.runner --experiment rabah-pruned-ctc/8-layer-int8

# Run new two-stage faster-whisper pipeline
.venv/bin/python -m benchmark.runner --experiment two-stage-faster-whisper-pruned

# Run NVIDIA FastConformer benchmark (requires: pip install -e .[nemo])
.venv/bin/python -m benchmark.runner --experiment nvidia-fastconformer

# Filter by audio category
.venv/bin/python -m benchmark.runner --category short
```

Results are saved to `benchmark/results/<timestamp>.json`. The runner also maintains `benchmark/results/latest.json` with best results per scoped run (`mode`, `category`, and sample count).

To build local Rabah pruned checkpoints (naive, no fine-tuning):

```bash
.venv/bin/python scripts/build_rabah_pruned_models.py --layers 12 8 6 --save-source
```

To fine-tune pruned models on Modal A100 GPU:

```bash
modal run --detach scripts/train_pruned_ctc_modal.py --layers 8 --strategy first_n
modal run scripts/train_pruned_ctc_modal.py --layers 8 --strategy first_n --download-only
```

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

2. **Two-stage retrieval works.** Using ASR only for candidate retrieval (top 50) then CTC for the final decision gets 72% SeqAcc -- only 9 points behind scoring all 6,236 verses. The retrieval step is fast and the accuracy gap comes from candidate recall, not re-scoring quality.

3. **ASR quality is the bottleneck for transcribe-then-match approaches.** All ASR-based approaches fail on the same samples. Two-stage sidesteps this by using ASR only for candidate retrieval, not the final decision.

4. **Layer pruning + fine-tuning works for CTC size reduction.** While there are no small Arabic wav2vec2 models pretrained from scratch, pruning a large Quran-specific model (24→8 layers) and fine-tuning the CTC head recovers most accuracy (72% vs 81% full model). The `first_n` strategy (keep contiguous early layers) outperforms `evenly_spaced` by 16 percentage points, suggesting early transformer layers carry the most transferable features.

5. **Contrastive audio-text matching needs a multilingual encoder.** Both embedding search (HuBERT, 0% cross-speaker) and contrastive-v2 (wav2vec2-base, 9% val accuracy) failed because English-pretrained audio encoders don't produce useful features for Arabic speech. Multi-reciter training and deeper projection heads aren't enough to overcome bad audio representations.

6. **Small ASR models can match large ones.** Moonshine Tiny Arabic (103 MB) matches Whisper Large-v3-Turbo (3.1 GB) on our benchmark. But fine-tuning Moonshine degrades it (LLaMA tokenizer with character-level Arabic tokens is fragile).

7. **Short verses are hard across all approaches.** Verses under 3-4 words don't provide enough signal. May need minimum-length gating or surah-context bias in the final product.

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

### wav2vec2-base Arabic CTC fine-tuning

Attempted to fine-tune `facebook/wav2vec2-base` (95M params, English SSL pretrained) with an Arabic CTC head, using the vocabulary from `jonatasgrosman/wav2vec2-large-xlsr-53-arabic`. Trained on EveryAyah + RetaSy datasets (85/15 interleave) on Modal A100-80GB.

**What we tried:**
- LR 3e-4, 5000 steps, batch 32 x grad_accum 2 → model collapsed at step ~1400, outputting only token 46 (a diacritic)
- LR 1e-4, 5000 steps → loss plateaued at 3.2-3.3 through all 5000 steps, model outputs only token 0 (pad)
- Frozen CNN feature extractor, only training transformer layers + CTC head

**Root cause:** wav2vec2-base was pretrained on English-only LibriSpeech. Its SSL representations encode English phonemes, not Arabic ones. Fine-tuning the transformer layers and CTC head alone can't bridge this gap -- the CNN feature extractor (which is frozen) produces features that don't represent Arabic speech sounds. The loss never drops below 3.2 because the model fundamentally can't distinguish Arabic characters from the audio features it receives.

**Alternatives considered:**
- `facebook/mms-300m` (multilingual, knows Arabic) -- 300M params, barely smaller than the 317M large model
- `facebook/wav2vec2-xls-r-300m` (multilingual) -- same size issue
- `DistilHuBERT` (23.5M params) -- English-only, same problem as wav2vec2-base

**Conclusion:** There is no existing small (<150M params) wav2vec2-family model with Arabic speech representations. Getting a small CTC model requires either (a) distilling from a multilingual model that already works, or (b) a different architecture entirely. This blocks both the distilled-ctc and two-stage experiments' path to the size target.

### Contrastive v2 (QuranCLAP v2) training

Attempted CLIP-style contrastive learning with wav2vec2-base (audio) + AraBERT v02 (text). Trained on 30k EveryAyah samples on Modal A100-80GB, batch 32 with gradient accumulation 4 (effective batch 128).

**Training log:**

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|-----------|-----------|----------|---------|
| P1 E1 (frozen) | 2.88 | 9.7% | 3.05 | 9.0% |
| P1 E6 (frozen) | 2.56 | 13.9% | 3.12 | 8.8% |
| P2 E1 (unfrozen) | 2.77 | 11.9% | 3.03 | 9.0% |
| P2 E2 (unfrozen) | 2.56 | 14.1% | 3.07 | 9.4% |

Random chance with batch 32 = 3.1%. Val accuracy plateaued at ~9% across all 8 epochs while train accuracy climbed to 14% -- classic overfitting. Unfreezing the last 2 encoder layers in Phase 2 didn't help.

**Root cause:** Same as CTC: wav2vec2-base produces English speech features, not Arabic ones. The projection heads can memorize training pairs but can't learn generalizable audio-text alignment because the audio features don't encode Arabic phonetic content. The AraBERT text encoder works fine -- the bottleneck is entirely on the audio side.

**Conclusion:** Any approach using wav2vec2-base as an Arabic audio encoder will fail. Future contrastive attempts need a multilingual audio encoder (e.g., XLS-R-300M, MMS-1B) or a completely different architecture.

## Further reading

- `REPORT.md` -- Full experiment report with per-sample breakdowns, failure analysis, and recommendations
- `RESEARCH-audio-to-verse.md` -- Research survey of approaches (WavLink, Moonshine v2, WhisperKit, contrastive learning, audio fingerprinting)
- `docs/plans/` -- Design documents for individual experiments
- Individual experiment `README.md` files for reproduction instructions
