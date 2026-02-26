"""
Knowledge distillation: large wav2vec2-xlsr-53-arabic → small wav2vec2-base CTC.

Teacher: jonatasgrosman/wav2vec2-large-xlsr-53-arabic (315M, 1.2GB)
Student: wav2vec2-base + Arabic CTC head (95M, ~380MB)

Loss = α * CTC_loss(student, labels) + (1-α) * KL(student || teacher)

Output: Modal volume → data/ctc-base-distilled/

Usage:
    modal run scripts/train_distill_modal.py
"""
import modal

app = modal.App("ctc-distill-quran")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "torch",
        "transformers>=4.40",
        "datasets>=3.0,<4.0",
        "accelerate",
        "soundfile",
        "librosa",
    )
)

vol = modal.Volume.from_name("ctc-quran-training", create_if_missing=True)


@app.function(
    image=image,
    gpu="A10G",
    timeout=10800,  # 3 hours
    volumes={"/training": vol},
    memory=32768,  # 32 GB RAM for both models
)
def train(alpha: float = 0.5, temperature: float = 2.0, max_steps: int = 5000):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from pathlib import Path
    from datasets import load_dataset, Audio, interleave_datasets
    from transformers import (
        Wav2Vec2ForCTC,
        Wav2Vec2Processor,
        Wav2Vec2Config,
        Wav2Vec2Model,
    )
    import numpy as np

    OUTPUT_DIR = Path("/training/ctc-base-distilled")
    CHECKPOINT_DIR = Path("/training/distill-checkpoints")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load teacher model ──
    print("Loading teacher model (wav2vec2-large-xlsr-53-arabic)...")
    teacher_processor = Wav2Vec2Processor.from_pretrained(
        "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"
    )
    teacher = Wav2Vec2ForCTC.from_pretrained(
        "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"
    )
    teacher.eval()
    teacher.to(device)
    for p in teacher.parameters():
        p.requires_grad = False

    vocab_size = teacher_processor.tokenizer.vocab_size
    pad_token_id = teacher_processor.tokenizer.pad_token_id

    # ── Create student model ──
    # Try to load pre-fine-tuned student if available
    student_pretrained = Path("/training/ctc-base-finetuned")
    if student_pretrained.exists():
        print(f"Loading pre-fine-tuned student from {student_pretrained}...")
        student = Wav2Vec2ForCTC.from_pretrained(str(student_pretrained))
    else:
        print("Creating fresh wav2vec2-base student with Arabic CTC head...")
        base_config = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-base")
        base_config.vocab_size = vocab_size
        base_config.ctc_loss_reduction = "mean"
        base_config.pad_token_id = pad_token_id
        base_config.ctc_zero_infinity = True
        student = Wav2Vec2ForCTC(base_config)
        pretrained = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        student.wav2vec2.load_state_dict(pretrained.state_dict())
        del pretrained

    student.freeze_feature_encoder()
    student.to(device)
    student.train()

    processor = teacher_processor

    print(f"Teacher: {sum(p.numel() for p in teacher.parameters()):,} params")
    print(f"Student: {sum(p.numel() for p in student.parameters()):,} params")
    print(f"  Trainable: {sum(p.numel() for p in student.parameters() if p.requires_grad):,}")
    print(f"Alpha: {alpha}, Temperature: {temperature}")

    # ── Load dataset ──
    print("Loading EveryAyah dataset (streaming)...")
    everyayah = load_dataset("tarteel-ai/everyayah", split="train", streaming=True)
    everyayah = everyayah.cast_column("audio", Audio(sampling_rate=16000))
    everyayah = everyayah.filter(lambda x: x["duration"] <= 20.0)  # shorter for distillation

    # ── Training loop ──
    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=1e-4,
        weight_decay=0.01,
    )

    ctc_loss_fn = nn.CTCLoss(blank=pad_token_id, reduction="mean", zero_infinity=True)

    print(f"\n{'='*60}")
    print(f"  Starting knowledge distillation ({max_steps} steps)")
    print(f"{'='*60}\n")

    step = 0
    running_loss = 0.0
    running_ctc = 0.0
    running_kl = 0.0

    for sample in everyayah:
        if step >= max_steps:
            break

        try:
            audio = sample["audio"]["array"]
            text = sample["text"]

            # Process audio
            inputs = processor.feature_extractor(
                audio, sampling_rate=16000, return_tensors="pt", padding=True
            )
            input_values = inputs.input_values.to(device)

            # Tokenize for CTC labels
            with processor.as_target_processor():
                labels = processor(text).input_ids
            labels_tensor = torch.tensor([labels], dtype=torch.long, device=device)
            label_lengths = torch.tensor([len(labels)], dtype=torch.long, device=device)

            # Teacher forward (no grad)
            with torch.no_grad():
                teacher_logits = teacher(input_values).logits  # (1, T_teacher, V)

            # Student forward
            student_out = student(input_values)
            student_logits = student_out.logits  # (1, T_student, V)

            T_student = student_logits.size(1)
            T_teacher = teacher_logits.size(1)

            # CTC loss on ground truth
            log_probs = F.log_softmax(student_logits, dim=-1).permute(1, 0, 2)  # (T, 1, V)
            input_lengths = torch.tensor([T_student], dtype=torch.long, device=device)
            ctc = ctc_loss_fn(log_probs, labels_tensor, input_lengths, label_lengths)

            # KL divergence on logit distributions
            # Align temporal dimensions (interpolate teacher to student length)
            if T_teacher != T_student:
                teacher_aligned = F.interpolate(
                    teacher_logits.permute(0, 2, 1),
                    size=T_student,
                    mode="linear",
                    align_corners=False,
                ).permute(0, 2, 1)
            else:
                teacher_aligned = teacher_logits

            student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
            teacher_probs = F.softmax(teacher_aligned / temperature, dim=-1)
            kl = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)

            # Combined loss
            loss = alpha * ctc + (1 - alpha) * kl

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            running_ctc += ctc.item()
            running_kl += kl.item()
            step += 1

            if step % 100 == 0:
                avg_loss = running_loss / 100
                avg_ctc = running_ctc / 100
                avg_kl = running_kl / 100
                print(f"Step {step}/{max_steps}: loss={avg_loss:.4f} (ctc={avg_ctc:.4f}, kl={avg_kl:.4f})")
                running_loss = 0.0
                running_ctc = 0.0
                running_kl = 0.0

            if step % 1000 == 0:
                ckpt_path = CHECKPOINT_DIR / f"step_{step}"
                student.save_pretrained(str(ckpt_path))
                vol.commit()
                print(f"  Saved checkpoint at step {step}")

        except Exception as e:
            print(f"  Skipping sample (error: {e})")
            continue

    # ── Save final model ──
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving distilled model to {OUTPUT_DIR}...")
    student.save_pretrained(str(OUTPUT_DIR))
    processor.save_pretrained(str(OUTPUT_DIR))
    vol.commit()

    print("Done! Distilled model saved to Modal volume.")


@app.function(
    image=image,
    volumes={"/training": vol},
)
def download_model():
    from pathlib import Path

    model_dir = Path("/training/ctc-base-distilled")
    if not model_dir.exists():
        print("No distilled model found! Run training first.")
        return {}

    files = {}
    for f in model_dir.rglob("*"):
        if f.is_file():
            rel = str(f.relative_to(model_dir))
            files[rel] = f.read_bytes()
            print(f"  {rel}: {len(files[rel]):,} bytes")
    return files


@app.local_entrypoint()
def main():
    from pathlib import Path

    print("Starting knowledge distillation on Modal GPU...")
    train.remote(alpha=0.5, temperature=2.0, max_steps=5000)

    print("\nDownloading distilled model...")
    out_dir = Path("data/ctc-base-distilled")
    out_dir.mkdir(parents=True, exist_ok=True)

    files = download_model.remote()
    for name, data in files.items():
        path = out_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        print(f"  Saved {name} ({len(data):,} bytes)")

    total = sum(len(d) for d in files.values())
    print(f"\nDistilled model saved to {out_dir} ({total / 1e6:.0f} MB total)")
