"""Test the streaming pipeline with text-based input."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.streaming import StreamingPipeline
from shared.quran_db import QuranDB

db = QuranDB()

def test_pipeline_single_verse():
    v = db.get_verse(1, 1)
    pipeline = StreamingPipeline(db=db)
    emissions = pipeline.run_on_text([v["text_clean"]])
    assert len(emissions) >= 1
    assert emissions[0]["surah"] == 1 and emissions[0]["ayah"] == 1

def test_pipeline_multi_verse_sequence():
    v1 = db.get_verse(103, 1)
    v2 = db.get_verse(103, 2)
    v3 = db.get_verse(103, 3)
    pipeline = StreamingPipeline(db=db)
    # Simulate accumulated text growing over time
    emissions = pipeline.run_on_text([
        v1["text_clean"],
        v1["text_clean"] + " " + v2["text_clean"],
        v1["text_clean"] + " " + v2["text_clean"] + " " + v3["text_clean"],
    ])
    assert len(emissions) >= 3
    assert emissions[0]["surah"] == 103 and emissions[0]["ayah"] == 1
    assert emissions[1]["surah"] == 103 and emissions[1]["ayah"] == 2
    assert emissions[2]["surah"] == 103 and emissions[2]["ayah"] == 3

def test_pipeline_run_on_full_transcript():
    """Test run_on_full_transcript with a mock transcribe function."""
    v = db.get_verse(112, 1)
    pipeline = StreamingPipeline(db=db)
    # Mock transcribe that returns the verse text
    mock_transcribe = lambda path: v["text_clean"]
    emissions = pipeline.run_on_full_transcript("dummy_path.wav", mock_transcribe)
    assert len(emissions) >= 1
    assert emissions[0]["surah"] == 112 and emissions[0]["ayah"] == 1
