import json
from pathlib import Path
from Levenshtein import ratio
from shared.normalizer import normalize_arabic

# Resolve to project root / data / quran.json
DATA_PATH = Path(__file__).parent.parent / "data" / "quran.json"


class QuranDB:
    def __init__(self, path: Path = DATA_PATH):
        with open(path) as f:
            self.verses = json.load(f)
        self._by_ref = {}
        self._by_surah = {}
        for v in self.verses:
            self._by_ref[(v["surah"], v["ayah"])] = v
            self._by_surah.setdefault(v["surah"], []).append(v)

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
            bonuses[(h_surah, h_ayah + 1)] = 0.15
            if self._by_ref.get((h_surah, h_ayah + 2)):
                bonuses[(h_surah, h_ayah + 2)] = 0.08
        else:
            # Last ayah in surah — bonus carries to first ayah(s) of next surah
            next_verses = self._by_surah.get(h_surah + 1, [])
            if next_verses:
                bonuses[(next_verses[0]["surah"], next_verses[0]["ayah"])] = 0.15
                if len(next_verses) > 1:
                    bonuses[(next_verses[1]["surah"], next_verses[1]["ayah"])] = 0.08
        return bonuses

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
            bonus = bonuses.get((v["surah"], v["ayah"]), 0.0)
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
                    combined = " ".join(c["text_clean"] for c in chunk)
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
