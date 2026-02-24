export interface VerseState {
  surah: number;
  ayah: number;
  surah_name_en: string;
  text_uthmani: string;
  words: string[];
  matched_indices: number[];
  word_position: number;
  total_words: number;
  confidence: number;
  progress_pct: number;
}

export interface CompletedVerse {
  surah: number;
  ayah: number;
  surah_name_en: string;
  text_uthmani: string;
}

export interface ServerMessage {
  type: "verse_update" | "status";
  current?: VerseState | null;
  completed?: CompletedVerse[];
  status?: string;
}
