import type { VerseState } from "../types";

interface Props {
  verse: VerseState;
}

export function CurrentVerse({ verse }: Props) {
  return (
    <div className="current-verse">
      <div className="verse-ref">
        <span className="verse-ref-surah">{verse.surah_name_en}</span>
        <span className="verse-ref-sep">&mdash;</span>
        <span className="verse-ref-num">
          {verse.surah}:{verse.ayah}
        </span>
        {verse.confidence > 0 && (
          <span className="verse-confidence">
            {Math.round(verse.confidence * 100)}%
          </span>
        )}
      </div>

      <div className="verse-words" dir="rtl" lang="ar">
        {verse.words.map((word, i) => {
          const isMatched = verse.matched_indices.includes(i);
          return (
            <span
              key={`${verse.surah}-${verse.ayah}-${i}`}
              className={`verse-word ${isMatched ? "matched" : "unmatched"}`}
            >
              {word}
            </span>
          );
        })}
      </div>

      <div className="verse-progress-track">
        <div
          className="verse-progress-fill"
          style={{ width: `${verse.progress_pct}%` }}
        />
      </div>
    </div>
  );
}
