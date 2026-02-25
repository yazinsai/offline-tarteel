"""
Streaming pipeline — connects chunked audio transcription to verse detection.

Can be used with any ASR backend that exposes a transcribe() function.
Supports three modes:
  - run_on_text(): test with pre-transcribed text chunks
  - run_on_full_transcript(): transcribe whole file, then detect verses
  - run_on_audio_chunked(): split audio into chunks, transcribe each,
    feed growing accumulated text to VerseTracker progressively
"""

import os
import tempfile

import numpy as np
import soundfile as sf

from shared.audio import load_audio
from shared.quran_db import QuranDB
from shared.verse_tracker import VerseTracker

SAMPLE_RATE = 16000
MIN_CHUNK_SAMPLES = 8000  # 0.5s — skip chunks shorter than this


class StreamingPipeline:
    """Streaming verse detection pipeline."""

    def __init__(self, db: QuranDB = None):
        self.db = db or QuranDB()

    def run_on_text(self, text_chunks: list[str]) -> list[dict]:
        """Run verse detection on a sequence of accumulated text snapshots.

        Args:
            text_chunks: List of accumulated transcripts (each chunk is the
                         full text so far, not a delta). This matches how
                         StreamingTranscriber.stream() yields accumulated_text.

        Returns:
            Ordered list of verse emissions [{"surah", "ayah", "score"}]
        """
        tracker = VerseTracker(self.db)
        all_emissions = []

        for text in text_chunks:
            emissions = tracker.process_text(text)
            all_emissions.extend(emissions)

        all_emissions.extend(tracker.finalize())
        return all_emissions

    def run_on_full_transcript(self, audio_path: str, transcribe_fn) -> list[dict]:
        """Run verse detection on a full transcript (non-streaming).

        Transcribes the whole file at once, then feeds to VerseTracker.
        Useful as a baseline and for backends that don't support chunking.

        Args:
            audio_path: Path to audio file
            transcribe_fn: Function(audio_path: str) -> str

        Returns:
            Ordered list of verse emissions [{"surah", "ayah", "score"}]
        """
        transcript = transcribe_fn(audio_path)
        tracker = VerseTracker(self.db)
        emissions = tracker.process_text(transcript)
        emissions += tracker.finalize()
        return emissions

    def run_on_audio_chunked(
        self,
        audio_path: str,
        transcribe_fn,
        chunk_seconds: float = 3.0,
        overlap_seconds: float = 0.0,
    ) -> list[dict]:
        """Run streaming verse detection with chunked audio.

        Splits audio into chunks, transcribes each independently using
        the provided backend, accumulates text, and feeds the growing
        transcript to VerseTracker after each chunk.

        Args:
            audio_path: Path to audio file
            transcribe_fn: Function(audio_path: str) -> str
            chunk_seconds: Duration of each audio chunk
            overlap_seconds: Overlap between consecutive chunks

        Returns:
            Ordered list of verse emissions [{"surah", "ayah", "score"}]
        """
        audio = load_audio(audio_path)
        chunk_size = int(chunk_seconds * SAMPLE_RATE)
        overlap_size = int(overlap_seconds * SAMPLE_RATE)
        step_size = max(chunk_size - overlap_size, 1)

        tracker = VerseTracker(self.db)
        all_emissions = []

        pos = 0
        while pos < len(audio):
            chunk_end = min(pos + chunk_size, len(audio))
            chunk = audio[pos:chunk_end]

            # Skip very short final chunks
            if len(chunk) < MIN_CHUNK_SAMPLES:
                break

            # Pad short chunks to 1s so Whisper-based models don't choke
            if len(chunk) < SAMPLE_RATE:
                chunk = np.pad(chunk, (0, SAMPLE_RATE - len(chunk)))

            # Save chunk to temp WAV, transcribe, clean up
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            try:
                sf.write(tmp.name, chunk, SAMPLE_RATE)
                tmp.close()
                chunk_text = transcribe_fn(tmp.name).strip()
            except Exception:
                chunk_text = ""
            finally:
                os.unlink(tmp.name)

            if chunk_text:
                # Use process_delta so text trimming from previous
                # emissions is preserved across chunks
                emissions = tracker.process_delta(chunk_text)
                all_emissions.extend(emissions)

            pos += step_size

        all_emissions.extend(tracker.finalize())
        return all_emissions
