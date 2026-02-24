import type { CompletedVerse } from "../types";

interface Props {
  verses: CompletedVerse[];
}

export function CompletedVerses({ verses }: Props) {
  if (verses.length === 0) return null;

  return (
    <div className="completed-verses">
      {verses.map((v, i) => (
        <div
          key={`${v.surah}-${v.ayah}-${i}`}
          className="completed-verse"
          style={{
            animationDelay: `${i * 60}ms`,
          }}
        >
          <div className="completed-ref">
            {v.surah_name_en} {v.surah}:{v.ayah}
          </div>
          <div className="completed-text" dir="rtl" lang="ar">
            {v.text_uthmani}
          </div>
        </div>
      ))}
    </div>
  );
}
