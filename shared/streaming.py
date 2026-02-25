"""
Streaming pipeline — connects chunked audio transcription to verse detection.

Can be used with any ASR backend that exposes a transcribe() function.
Also provides run_on_text() for testing with pre-transcribed text chunks.
"""

from shared.quran_db import QuranDB
from shared.verse_tracker import VerseTracker


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
