export class CTCDecoder {
  private vocab: Map<number, string>;
  private blankId: number;

  constructor(vocabJson: Record<string, string>) {
    this.vocab = new Map();
    let maxId = 0;
    for (const [id, token] of Object.entries(vocabJson)) {
      const numId = parseInt(id);
      this.vocab.set(numId, token);
      if (numId > maxId) maxId = numId;
    }
    this.blankId = maxId; // blank is the last token
  }

  decode(logprobs: Float32Array, timeSteps: number, vocabSize: number): string {
    // argmax per timestep
    const ids: number[] = [];
    for (let t = 0; t < timeSteps; t++) {
      let maxIdx = 0;
      let maxVal = logprobs[t * vocabSize];
      for (let v = 1; v < vocabSize; v++) {
        const val = logprobs[t * vocabSize + v];
        if (val > maxVal) {
          maxVal = val;
          maxIdx = v;
        }
      }
      ids.push(maxIdx);
    }

    // Collapse consecutive duplicates, remove blanks
    const tokens: string[] = [];
    let prev = -1;
    for (const id of ids) {
      if (id !== prev && id !== this.blankId) {
        const token = this.vocab.get(id) ?? "";
        tokens.push(token);
      }
      prev = id;
    }

    // BPE detokenize: join tokens, replace ▁ (sentencepiece) with space
    return tokens.join("").replace(/▁/g, " ").trim();
  }
}
