"""
Benchmark runner for all experiments.

Usage:
    python -m benchmark.runner                           # all experiments
    python -m benchmark.runner --experiment whisper-lora  # one experiment
    python -m benchmark.runner --category short           # filter by category
"""

import sys
import json
import time
import importlib.util
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
CORPUS_DIR = Path(__file__).parent / "test_corpus"
RESULTS_DIR = Path(__file__).parent / "results"

# Experiments and their run.py paths
EXPERIMENT_REGISTRY = {
    "whisper-lora": EXPERIMENTS_DIR / "whisper-lora" / "run.py",
    "embedding-search": EXPERIMENTS_DIR / "embedding-search" / "run.py",
    "contrastive": EXPERIMENTS_DIR / "contrastive" / "run.py",
    "streaming-asr": EXPERIMENTS_DIR / "streaming-asr" / "run.py",
    "ctc-alignment": EXPERIMENTS_DIR / "ctc-alignment" / "run.py",
    "tarteel-whisper-base": EXPERIMENTS_DIR / "tarteel-whisper-base" / "run.py",
}

NEW_MODELS_PATH = EXPERIMENTS_DIR / "new-models" / "run.py"


def _load_module(name: str, file_path: Path):
    """Load a Python module from a file path (handles hyphenated dirs)."""
    spec = importlib.util.spec_from_file_location(name, str(file_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_manifest() -> list[dict]:
    manifest_path = CORPUS_DIR / "manifest.json"
    with open(manifest_path) as f:
        data = json.load(f)
    return data["samples"]


def discover_experiments(filter_name: str | None = None) -> list[dict]:
    """Return list of {name, module_path, model_name (optional)}."""
    experiments = []

    for name, run_path in EXPERIMENT_REGISTRY.items():
        if filter_name and filter_name != name:
            continue
        if not run_path.exists():
            print(f"Warning: {name} run.py not found at {run_path}")
            continue
        experiments.append({"name": name, "run_path": run_path, "model_name": None})

    # Expand new-models
    if not filter_name or filter_name.startswith("new-models"):
        try:
            mod = _load_module("new_models_run", NEW_MODELS_PATH)
            for model_name in mod.list_models():
                entry_name = f"new-models/{model_name}"
                if filter_name and filter_name != entry_name and filter_name != "new-models":
                    continue
                experiments.append({
                    "name": entry_name,
                    "run_path": NEW_MODELS_PATH,
                    "model_name": model_name,
                })
        except Exception as e:
            print(f"Warning: could not load new-models: {e}")

    return experiments


def run_experiment(exp: dict, samples: list[dict]) -> dict:
    """Run one experiment against all samples. Returns results dict."""
    mod = _load_module(exp["name"].replace("/", "_").replace("-", "_"), exp["run_path"])

    # Warmup call
    warmup_sample = samples[0]
    audio_path = str(CORPUS_DIR / warmup_sample["file"])
    try:
        if exp["model_name"]:
            mod.predict(audio_path, model_name=exp["model_name"])
        else:
            mod.predict(audio_path)
    except Exception as e:
        print(f"  Warmup failed for {exp['name']}: {e}")

    # Get model size
    try:
        if exp["model_name"]:
            size = mod.model_size(model_name=exp["model_name"])
        else:
            size = mod.model_size()
    except Exception:
        size = 0

    correct = 0
    total = len(samples)
    latencies = []
    per_sample = []

    for sample in samples:
        audio_path = str(CORPUS_DIR / sample["file"])
        try:
            start = time.perf_counter()
            if exp["model_name"]:
                result = mod.predict(audio_path, model_name=exp["model_name"])
            else:
                result = mod.predict(audio_path)
            elapsed = time.perf_counter() - start
        except Exception as e:
            result = {"surah": 0, "ayah": 0, "ayah_end": None, "score": 0.0, "transcript": f"ERROR: {e}"}
            elapsed = 0.0

        expected = (sample["surah"], sample["ayah"], sample.get("ayah_end"))
        predicted = (result["surah"], result["ayah"], result.get("ayah_end"))
        is_correct = expected == predicted

        if is_correct:
            correct += 1
        latencies.append(elapsed)

        per_sample.append({
            "id": sample["id"],
            "expected": {"surah": sample["surah"], "ayah": sample["ayah"], "ayah_end": sample.get("ayah_end")},
            "predicted": result,
            "correct": is_correct,
            "latency": elapsed,
        })

    avg_latency = sum(latencies) / len(latencies) if latencies else 0

    return {
        "name": exp["name"],
        "accuracy": correct / total if total > 0 else 0,
        "correct": correct,
        "total": total,
        "avg_latency": avg_latency,
        "model_size": size,
        "per_sample": per_sample,
    }


def format_size(size_bytes: int) -> str:
    if size_bytes >= 1024 * 1024 * 1024:
        return f"{size_bytes / (1024**3):.1f} GB"
    return f"{size_bytes / (1024**2):.0f} MB"


def print_table(results: list[dict]):
    print()
    print(f"{'Experiment':<30} {'Accuracy':>12} {'Latency':>10} {'Model Size':>12}")
    print("-" * 66)
    for r in results:
        acc = f"{r['accuracy']:.0%} ({r['correct']}/{r['total']})"
        lat = f"{r['avg_latency']:.2f}s"
        size = format_size(r['model_size'])
        print(f"{r['name']:<30} {acc:>12} {lat:>10} {size:>12}")
    print()


def save_results(results: list[dict]):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    path = RESULTS_DIR / f"{timestamp}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {path}")

    # Update latest.json — merge new results with existing best-per-experiment
    latest_path = RESULTS_DIR / "latest.json"
    if latest_path.exists():
        with open(latest_path) as f:
            latest = {r["name"]: r for r in json.load(f)}
    else:
        latest = {}

    for r in results:
        summary = {
            "name": r["name"],
            "accuracy": r["accuracy"],
            "correct": r["correct"],
            "total": r["total"],
            "avg_latency": r["avg_latency"],
            "model_size": r["model_size"],
            "timestamp": timestamp,
        }
        # Keep the better result (higher accuracy, or same accuracy but lower latency)
        prev = latest.get(r["name"])
        if prev is None or r["accuracy"] > prev["accuracy"] or (
            r["accuracy"] == prev["accuracy"] and r["avg_latency"] < prev.get("avg_latency", float("inf"))
        ):
            latest[r["name"]] = summary

    with open(latest_path, "w") as f:
        json.dump(sorted(latest.values(), key=lambda x: x["name"]), f, indent=2, default=str)
    print(f"Updated {latest_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark all experiments")
    parser.add_argument("--experiment", type=str, help="Run only this experiment")
    parser.add_argument("--category", type=str, help="Filter samples by category")
    args = parser.parse_args()

    samples = load_manifest()
    if args.category:
        samples = [s for s in samples if s["category"] == args.category]
        print(f"Filtered to {len(samples)} samples in category '{args.category}'")

    experiments = discover_experiments(args.experiment)
    if not experiments:
        print(f"No experiments found matching '{args.experiment}'")
        return

    print(f"Running {len(experiments)} experiment(s) on {len(samples)} sample(s)...")

    results = []
    for exp in experiments:
        print(f"\n>>> {exp['name']}")
        result = run_experiment(exp, samples)
        results.append(result)
        print(f"    Accuracy: {result['accuracy']:.0%} ({result['correct']}/{result['total']})")

    print_table(results)
    save_results(results)


if __name__ == "__main__":
    main()
