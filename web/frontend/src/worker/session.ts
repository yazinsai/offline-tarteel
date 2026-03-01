import * as ort from "onnxruntime-web";

let session: ort.InferenceSession | null = null;

export async function createSession(modelBuffer: ArrayBuffer): Promise<void> {
  // Try WebGPU first, fall back to WASM
  const providers: string[] = [];
  if (typeof navigator !== "undefined" && "gpu" in navigator) {
    try {
      const adapter = await (navigator as any).gpu.requestAdapter();
      if (adapter) providers.push("webgpu");
    } catch {
      // WebGPU not available
    }
  }
  providers.push("wasm");

  ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;
  ort.env.wasm.simd = true;

  session = await ort.InferenceSession.create(modelBuffer, {
    executionProviders: providers,
  });
}

export async function runInference(
  melFeatures: Float32Array,
  numMels: number,
  timeFrames: number,
): Promise<{ logprobs: Float32Array; timeSteps: number; vocabSize: number }> {
  if (!session) throw new Error("Session not initialized");

  const inputTensor = new ort.Tensor("float32", melFeatures, [
    1,
    numMels,
    timeFrames,
  ]);
  const lengthTensor = new ort.Tensor(
    "int64",
    BigInt64Array.from([BigInt(timeFrames)]),
    [1],
  );

  const inputNames = session.inputNames;
  const feeds: Record<string, ort.Tensor> = {
    [inputNames[0]]: inputTensor,
    [inputNames[1]]: lengthTensor,
  };

  const results = await session.run(feeds);
  const outputTensor = results[session.outputNames[0]];
  const [_batch, timeSteps, vocabSize] = outputTensor.dims as number[];

  return {
    logprobs: outputTensor.data as Float32Array,
    timeSteps,
    vocabSize,
  };
}
