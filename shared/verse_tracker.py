"""
VerseTracker — Detect verse boundaries in streaming Quran transcription.

As accumulated text grows, matches against QuranDB and emits (surah, ayah)
when a verse is "complete" (match shifts to a different verse or score drops).
Continuation bias resolves duplicate-text ambiguity by preferring the next
sequential verse after the last emission.
"""

from Levenshtein import ratio
from shared.normalizer import normalize_arabic
from shared.quran_db import QuranDB

CONTINUATION_BONUS = 0.15
SCORE_DROP_THRESHOLD = 0.15  # emit when score drops this much below peak
MIN_EMIT_SCORE = 0.3  # minimum score to consider a match real
OVERFLOW_RATIO = 1.15  # if accumulated words > verse words * this, try splitting


class VerseTracker:
    """Track and emit verse detections from streaming text."""

    def __init__(
        self,
        db: QuranDB = None,
        last_emission: tuple[int, int] | None = None,
    ):
        self.db = db or QuranDB()
        self._accumulated = ""
        self._current_match: dict | None = None  # {surah, ayah, text_clean, score}
        self._peak_score: float = 0.0
        self._emissions: list[dict] = []
        self._last_emitted: tuple[int, int] | None = last_emission

    def _score_verse(self, text: str, verse_clean: str, surah: int, ayah: int) -> float:
        """Score a candidate verse against text, with prefix-awareness and continuation bias."""
        text_words = text.split()
        verse_words = verse_clean.split()
        n_text = len(text_words)
        n_verse = len(verse_words)

        # Prefix scoring: compare text against the first N words of the verse
        prefix_len = min(n_text, n_verse)
        prefix = " ".join(verse_words[:prefix_len])
        prefix_score = ratio(text, prefix)
        full_score = ratio(text, verse_clean)

        coverage = n_text / max(n_verse, 1)
        if coverage > 0.8:
            raw = 0.3 * prefix_score + 0.7 * full_score
        else:
            raw = 0.7 * prefix_score + 0.3 * full_score

        # Continuation bias: boost the next expected verse
        if self._last_emitted:
            next_v = self.db.get_next_verse(*self._last_emitted)
            if next_v and next_v["surah"] == surah and next_v["ayah"] == ayah:
                raw += CONTINUATION_BONUS

        return raw

    def _find_best_match(self, text: str) -> dict | None:
        """Find the best matching single verse for the given text."""
        if not text.strip():
            return None

        best = None
        best_score = 0.0

        for v in self.db.verses:
            score = self._score_verse(text, v["text_clean"], v["surah"], v["ayah"])
            if score > best_score:
                best_score = score
                best = {
                    "surah": v["surah"],
                    "ayah": v["ayah"],
                    "text_clean": v["text_clean"],
                    "score": best_score,
                }

        if best and best["score"] >= MIN_EMIT_SCORE:
            return best
        return None

    def _emit(self, match: dict) -> dict:
        """Emit a verse detection and reset state."""
        emission = {"surah": match["surah"], "ayah": match["ayah"], "score": match["score"]}
        self._emissions.append(emission)
        self._last_emitted = (match["surah"], match["ayah"])

        # Trim accumulated text: remove the portion that matched
        matched_words = match["text_clean"].split()
        acc_words = self._accumulated.split()
        overlap = min(len(matched_words), len(acc_words))
        remaining_words = acc_words[overlap:]
        self._accumulated = " ".join(remaining_words)

        self._current_match = None
        self._peak_score = 0.0

        return emission

    def _try_split_and_emit(self, match: dict) -> list[dict]:
        """If accumulated text overflows the matched verse, emit and recurse on remainder."""
        emissions = []
        acc_words = self._accumulated.split()
        verse_words = match["text_clean"].split()

        if len(acc_words) > len(verse_words) * OVERFLOW_RATIO and len(verse_words) > 0:
            # Text is significantly longer than the best-matched verse.
            # Emit the match and try to match the remainder.
            emissions.append(self._emit(match))
            if self._accumulated.strip():
                next_match = self._find_best_match(self._accumulated)
                if next_match:
                    # Recursively check if the remainder also overflows
                    more = self._try_split_and_emit(next_match)
                    if more:
                        emissions.extend(more)
                    else:
                        self._current_match = next_match
                        self._peak_score = next_match["score"]

        return emissions

    def process_text(self, text: str) -> list[dict]:
        """Process new text, return any verse emissions.

        Args:
            text: The full accumulated transcript so far (not a delta).

        Returns:
            List of emitted verses [{"surah": int, "ayah": int, "score": float}]
        """
        normalized = normalize_arabic(text)
        if not normalized.strip():
            return []

        self._accumulated = normalized
        emissions = []

        match = self._find_best_match(self._accumulated)
        if not match:
            return []

        same_verse = (
            self._current_match
            and self._current_match["surah"] == match["surah"]
            and self._current_match["ayah"] == match["ayah"]
        )

        if same_verse:
            # Same verse — update peak score
            if match["score"] > self._peak_score:
                self._peak_score = match["score"]
            elif self._peak_score - match["score"] > SCORE_DROP_THRESHOLD:
                # Score dropped — verse is likely complete, emit it
                emissions.append(self._emit(self._current_match))
                # Try to match remainder
                if self._accumulated.strip():
                    next_match = self._find_best_match(self._accumulated)
                    if next_match:
                        self._current_match = next_match
                        self._peak_score = next_match["score"]
                    else:
                        self._current_match = None
                        self._peak_score = 0.0
            else:
                self._current_match = match
        else:
            # Different verse detected
            if self._current_match and self._current_match["score"] >= MIN_EMIT_SCORE:
                # Emit the previous verse
                emissions.append(self._emit(self._current_match))
            # Start tracking the new verse
            self._current_match = match
            self._peak_score = match["score"]

        if not self._current_match:
            self._current_match = match
            self._peak_score = match["score"]

        # Check if accumulated text overflows the current match — if so, split
        if self._current_match and not emissions:
            split_emissions = self._try_split_and_emit(self._current_match)
            if split_emissions:
                emissions.extend(split_emissions)

        return emissions

    def finalize(self) -> list[dict]:
        """Flush any remaining match as a final emission."""
        if self._current_match and self._current_match["score"] >= MIN_EMIT_SCORE:
            return [self._emit(self._current_match)]
        return []
