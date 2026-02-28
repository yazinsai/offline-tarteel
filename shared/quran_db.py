import json
from pathlib import Path
from Levenshtein import ratio
from shared.normalizer import normalize_arabic


def partial_ratio(short: str, long: str) -> float:
    """Levenshtein ratio of *short* against its best-matching window in *long*.

    Useful for detecting when a short transcription is a fragment of a longer
    verse that was already emitted.
    """
    if not short or not long:
        return 0.0
    if len(short) > len(long):
        short, long = long, short
    window = len(short)
    best = 0.0
    for i in range(max(1, len(long) - window + 1)):
        r = ratio(short, long[i:i + window])
        if r > best:
            best = r
            if best == 1.0:
                break
    return best

# Resolve to project root / data / quran.json
DATA_PATH = Path(__file__).parent.parent / "data" / "quran.json"


_BSM_CLEAN = normalize_arabic("بسم الله الرحمن الرحيم")


class QuranDB:
    def __init__(self, path: Path = DATA_PATH):
        with open(path) as f:
            self.verses = json.load(f)
        self._by_ref = {}
        self._by_surah = {}
        for v in self.verses:
            self._by_ref[(v["surah"], v["ayah"])] = v
            self._by_surah.setdefault(v["surah"], []).append(v)
            # Pre-compute bismillah-stripped text for verse 1 of each surah
            # (Al-Fatiha 1:1 IS the bismillah, At-Tawbah 9 has none)
            if (
                v["ayah"] == 1
                and v["surah"] not in (1, 9)
                and v["text_clean"].startswith(_BSM_CLEAN)
            ):
                stripped = v["text_clean"][len(_BSM_CLEAN):].strip()
                v["text_clean_no_bsm"] = stripped if stripped else None
            else:
                v["text_clean_no_bsm"] = None

    @property
    def total_verses(self):
        return len(self.verses)

    @property
    def surah_count(self):
        return len(self._by_surah)

    def get_verse(self, surah: int, ayah: int):
        return self._by_ref.get((surah, ayah))

    def get_surah(self, surah: int):
        return self._by_surah.get(surah, [])

    def get_next_verse(self, surah: int, ayah: int) -> dict | None:
        """Return the next verse after surah:ayah, or None if last verse."""
        verses = self._by_surah.get(surah, [])
        for i, v in enumerate(verses):
            if v["ayah"] == ayah:
                if i + 1 < len(verses):
                    return verses[i + 1]
                next_surah = self._by_surah.get(surah + 1, [])
                return next_surah[0] if next_surah else None
        return None

    def search(self, text: str, top_k: int = 5) -> list[dict]:
        text = normalize_arabic(text)
        scored = []
        for v in self.verses:
            score = ratio(text, v["text_clean"])
            scored.append({**v, "score": score, "text": v["text_uthmani"]})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def _continuation_bonuses(
        self, hint: tuple[int, int] | None
    ) -> dict[tuple[int, int], float]:
        """Build a map of (surah, ayah) → score bonus for expected next verses."""
        if not hint:
            return {}
        h_surah, h_ayah = hint
        bonuses: dict[tuple[int, int], float] = {}
        nv = self._by_ref.get((h_surah, h_ayah + 1))
        if nv:
            bonuses[(h_surah, h_ayah + 1)] = 0.22
            if self._by_ref.get((h_surah, h_ayah + 2)):
                bonuses[(h_surah, h_ayah + 2)] = 0.12
            if self._by_ref.get((h_surah, h_ayah + 3)):
                bonuses[(h_surah, h_ayah + 3)] = 0.06
        else:
            # Last ayah in surah — bonus carries to first ayah(s) of next surah
            next_verses = self._by_surah.get(h_surah + 1, [])
            for i, nv in enumerate(next_verses[:3]):
                bonus = [0.22, 0.12, 0.06][i]
                bonuses[(nv["surah"], nv["ayah"])] = bonus
        return bonuses

    @staticmethod
    def _suffix_prefix_score(text: str, verse_text: str) -> float:
        """Best Levenshtein ratio from matching suffixes of *text* against
        equal-length prefixes of *verse_text*.

        After a window reset the transcription often starts with residual
        words from the *previous* verse followed by the start of the *next*
        verse. This method finds the best alignment by sliding the split
        point through the transcription.
        """
        words_t = text.split()
        words_v = verse_text.split()
        if len(words_t) < 2 or len(words_v) < 2:
            return 0.0
        best = 0.0
        max_trim = min(len(words_t) // 2, 4)
        for trim in range(1, max_trim + 1):
            suffix = " ".join(words_t[trim:])
            n = len(words_t) - trim
            prefix = " ".join(words_v[:min(n, len(words_v))])
            best = max(best, ratio(suffix, prefix))
        return best

    def match_verse(
        self,
        text: str,
        threshold: float = 0.3,
        max_span: int = 3,
        hint: tuple[int, int] | None = None,
        return_top_k: int = 0,
    ) -> dict | None:
        """Find the best matching verse or consecutive verse span.

        Two-pass: first find top single-verse candidates (fast), then try
        multi-ayah spans only around those candidates.

        If *hint* is provided as (surah, ayah) of the last matched verse,
        the expected next verses receive a score bonus so sequential
        recitation is favoured over re-inferring from scratch.

        If *return_top_k* > 0, the returned dict includes a ``"runners_up"``
        list with the next-best candidates (each with raw_score and bonus).
        """
        text = normalize_arabic(text)
        if not text.strip():
            return None

        bonuses = self._continuation_bonuses(hint)

        # Pass 1: score all single verses (with continuation bonus)
        scored = []
        for v in self.verses:
            raw = ratio(text, v["text_clean"])
            # Also try matching without the bismillah prefix for verse 1s
            if v["text_clean_no_bsm"]:
                raw = max(raw, ratio(text, v["text_clean_no_bsm"]))
            bonus = bonuses.get((v["surah"], v["ayah"]), 0.0)
            # For continuation candidates, also try suffix-prefix matching
            # to handle residual text from the previous verse in the window
            if bonus > 0:
                sp = self._suffix_prefix_score(text, v["text_clean"])
                raw = max(raw, sp)
            scored.append((v, raw, bonus, min(raw + bonus, 1.0)))
        scored.sort(key=lambda x: x[3], reverse=True)

        best_v, best_raw, best_bonus, best_score = scored[0]
        best = {**best_v, "score": best_score, "raw_score": best_raw, "bonus": best_bonus}

        # Collect single-verse runners-up before span pass
        top_singles = [
            {
                "surah": v["surah"],
                "ayah": v["ayah"],
                "raw_score": round(raw, 3),
                "bonus": round(bon, 3),
                "score": round(total, 3),
                "text_clean": v["text_clean"][:60],
            }
            for v, raw, bon, total in scored[:max(return_top_k, 5)]
        ]

        # Pass 2: try multi-ayah spans around top 20 candidates
        seen_surahs = set()
        for v, _raw, _bon, _total in scored[:20]:
            s = v["surah"]
            if s in seen_surahs:
                continue
            seen_surahs.add(s)
            verses = self._by_surah[s]
            for i, sv in enumerate(verses):
                for span in range(2, max_span + 1):
                    if i + span > len(verses):
                        break
                    chunk = verses[i:i + span]
                    # Use no-bismillah text for the first verse in a span
                    first_text = (
                        chunk[0]["text_clean_no_bsm"] or chunk[0]["text_clean"]
                    )
                    combined = " ".join(
                        [first_text] + [c["text_clean"] for c in chunk[1:]]
                    )
                    raw = ratio(text, combined)
                    bonus = bonuses.get((chunk[0]["surah"], chunk[0]["ayah"]), 0.0)
                    score = min(raw + bonus, 1.0)
                    if score > best_score:
                        best_score = score
                        best = {
                            "surah": s,
                            "ayah": chunk[0]["ayah"],
                            "ayah_end": chunk[-1]["ayah"],
                            "text": " ".join(c["text_uthmani"] for c in chunk),
                            "text_clean": combined,
                            "score": score,
                            "raw_score": raw,
                            "bonus": bonus,
                        }

        if best_score >= threshold:
            if return_top_k > 0:
                best["runners_up"] = top_singles[:return_top_k]
            return best
        return None
