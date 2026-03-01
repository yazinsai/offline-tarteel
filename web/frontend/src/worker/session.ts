import * as ort from "onnxruntime-web/wasm";

let session: ort.InferenceSession | null = null;

export async function createSession(modelBuffer: ArrayBuffer): Promise<void> {
  // Configure WASM paths — files are copied to public/ via postinstall
  ort.env.wasm.wasmPaths = "/";
  ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;
  ort.env.wasm.simd = true;

  // Use WASM backend (WebGPU support in onnxruntime-web is still experimental)
  const providers: string[] = ["wasm"];

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
