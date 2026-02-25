"""
Train wav2vec2 CTC model on Quran audio (Modal A10G).

Base model: facebook/wav2vec2-xls-r-300m
Dataset: Buraaq/quran-md-ayahs (187K samples, 30 reciters)
Approach: Freeze CNN feature extractor, fine-tune transformer + CTC head
          with custom Arabic vocabulary (normalized, no diacritics).

Usage:
    modal run scripts/train_ctc_modal.py
"""
import modal

app = modal.App("wav2vec2-quran-ctc")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1-dev")
    .pip_install(
        "torch",
        "transformers>=4.40",
        "datasets>=3.0,<4.0",
        "accelerate",
        "evaluate",
        "jiwer",
        "soundfile",
        "librosa",
    )
)

vol = modal.Volume.from_name("wav2vec2-quran-ctc", create_if_missing=True)

# Arabic characters after normalization (shared.normalizer compatible)
VOCAB_CHARS = list("ابتثجحخدذرزسشصضطظعغفقكلمنهويء ")
# Build vocab: <pad>=0, <s>=1, </s>=2, <unk>=3, | (word boundary)=4, then chars
VOCAB = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, "|": 4}
for i, c in enumerate(VOCAB_CHARS):
    VOCAB[c] = i + 5


@app.function(
    image=image,
    gpu="A10G",
    timeout=10800,  # 3 hours
    volumes={"/training": vol},
)
def train():
    import json
    import re
    import torch
    from pathlib import Path
    from dataclasses import dataclass
    from typing import Any, Union
    from datasets import load_dataset, Audio
    from transformers import (
        Wav2Vec2ForCTC,
        Wav2Vec2CTCTokenizer,
        Wav2Vec2FeatureExtractor,
        Wav2Vec2Processor,
        TrainingArguments,
        Trainer,
    )

    OUTPUT_DIR = Path("/training/ctc-model")
    CHECKPOINT_DIR = Path("/training/ctc-checkpoints")

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # -- Build vocab + tokenizer --
    vocab_path = Path("/tmp/vocab.json")
    vocab_path.write_text(json.dumps(VOCAB))

    tokenizer = Wav2Vec2CTCTokenizer(
        str(vocab_path),
        unk_token="<unk>",
        pad_token="<pad>",
        word_delimiter_token="|",
    )

    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )

    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
    )

    # -- Load model --
    print("Loading wav2vec2-xls-r-300m...")
    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-xls-r-300m",
        attention_dropout=0.1,
        hidden_dropout=0.1,
        feat_proj_dropout=0.0,
        mask_time_prob=0.05,
        layerdrop=0.1,
        ctc_loss_reduction="mean",
        pad_token_id=tokenizer.pad_token_id,
        vocab_size=len(tokenizer),
        ctc_zero_infinity=True,
    )

    # Freeze feature extractor (CNN layers)
    model.freeze_feature_encoder()

    # -- Arabic normalization (matches shared.normalizer) --
    _DIACRITICS = re.compile(
        '[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC'
        '\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED\u0640]'
    )
    _NORM_MAP = str.maketrans({
        '\u0623': '\u0627',  # أ -> ا
        '\u0625': '\u0627',  # إ -> ا
        '\u0622': '\u0627',  # آ -> ا
        '\u0671': '\u0627',  # ٱ -> ا
        '\u0629': '\u0647',  # ة -> ه
        '\u0649': '\u064A',  # ى -> ي
    })

    def normalize_arabic(text):
        text = _DIACRITICS.sub('', text)
        text = text.translate(_NORM_MAP)
        text = ' '.join(text.split())
        return text

    # -- Load dataset --
    print("Loading Buraaq/quran-md-ayahs...")
    ds = load_dataset("Buraaq/quran-md-ayahs", split="train", streaming=True)
    # Disable auto-decode — soundfile can't handle all audio formats in this dataset.
    # We'll decode manually with torchaudio (which uses ffmpeg) in the prepare function.
    ds = ds.cast_column("audio", Audio(decode=False))

    # Hold out 3 reciters for validation
    VAL_RECITERS = {
        "saood_ash_shuraym",
        "abdulbaset_abdulsamad_mojawwad",
        "hani_ar_rifai",
    }

    def is_train(example):
        return example["reciter_id"] not in VAL_RECITERS

    def is_val(example):
        return example["reciter_id"] in VAL_RECITERS

    train_ds = ds.filter(is_train)
    # For val, take a finite subset (streaming dataset)
    val_ds = ds.filter(is_val).take(2000)

    _skip_count = [0]

    def prepare(batch):
        import io
        import librosa
        import numpy as np

        try:
            # Decode audio with librosa (handles mp3 via audioread/ffmpeg)
            audio_data = batch["audio"]  # dict with 'bytes' and 'path'
            audio_bytes = audio_data["bytes"]
            audio_array, _ = librosa.load(
                io.BytesIO(audio_bytes), sr=16000, mono=True
            )

            # Skip very long audio (>15s to prevent OOM)
            if len(audio_array) > 16000 * 15:
                raise ValueError(f"Audio too long: {len(audio_array)/16000:.1f}s")

            input_values = processor(
                audio_array,
                sampling_rate=16000,
                return_tensors=None,
            ).input_values[0]

            # Normalize Arabic text (strip diacritics, normalize chars)
            text = normalize_arabic(batch["ayah_ar"])
            # Replace spaces with word delimiter for CTC
            text_with_delim = text.replace(" ", "|")
            labels = processor.tokenizer(text_with_delim).input_ids

            if len(labels) == 0:
                raise ValueError("Empty label sequence")

            return {"input_values": input_values, "labels": labels, "_valid": True}
        except Exception as e:
            _skip_count[0] += 1
            if _skip_count[0] <= 10:
                print(f"Skipping bad sample ({_skip_count[0]}): {e}")
            return {"input_values": [0.0], "labels": [0], "_valid": False}

    all_columns = [
        "surah_id", "ayah_id", "surah_name_ar", "surah_name_en",
        "surah_name_tr", "ayah_count", "ayah_ar", "ayah_en", "ayah_tr",
        "reciter_id", "reciter_name", "audio",
    ]
    train_ds = train_ds.map(prepare, remove_columns=all_columns)
    train_ds = train_ds.filter(lambda x: x["_valid"])
    train_ds = train_ds.remove_columns(["_valid"])

    # -- Data collator --
    @dataclass
    class DataCollatorCTCWithPadding:
        processor: Any
        padding: Union[bool, str] = True

        def __call__(self, features):
            input_values = [{"input_values": f["input_values"]} for f in features]
            batch = self.processor.feature_extractor.pad(
                input_values, padding=self.padding, return_tensors="pt"
            )

            labels = [f["labels"] for f in features]
            max_len = max(len(l) for l in labels)
            padded = [l + [-100] * (max_len - len(l)) for l in labels]
            batch["labels"] = torch.tensor(padded)
            return batch

    data_collator = DataCollatorCTCWithPadding(processor=processor)

    # -- Training --
    model.gradient_checkpointing_enable()

    training_args = TrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=3e-5,
        warmup_steps=500,
        max_steps=5000,
        fp16=True,
        logging_steps=50,
        save_steps=1000,
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=data_collator,
        processing_class=processor.feature_extractor,
    )

    print("\n" + "=" * 60)
    print("  Starting CTC training (5000 steps on A10G)")
    print("  Base: wav2vec2-xls-r-300m")
    print("  Dataset: Buraaq/quran-md-ayahs (30 reciters)")
    print("=" * 60 + "\n")

    trainer.train()

    # -- Save --
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving model to {OUTPUT_DIR}...")
    model.save_pretrained(str(OUTPUT_DIR))
    processor.save_pretrained(str(OUTPUT_DIR))
    vol.commit()
    print("Done!")


@app.function(image=image, volumes={"/training": vol})
def download_model():
    from pathlib import Path

    model_dir = Path("/training/ctc-model")
    if not model_dir.exists():
        print("No model found! Run training first.")
        return {}

    files = {}
    for f in model_dir.rglob("*"):
        if f.is_file():
            rel = str(f.relative_to(model_dir))
            files[rel] = f.read_bytes()
            size_mb = len(files[rel]) / (1024 * 1024)
            print(f"  {rel}: {size_mb:.1f} MB")
    return files


@app.local_entrypoint()
def main():
    from pathlib import Path

    print("Starting CTC training on Modal GPU...")
    train.remote()

    print("\nDownloading model...")
    out_dir = Path("data/ctc-model")
    out_dir.mkdir(parents=True, exist_ok=True)

    files = download_model.remote()
    for name, data in files.items():
        path = out_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        print(f"  Saved {name} ({len(data):,} bytes)")

    print(f"\nModel saved to {out_dir}")
