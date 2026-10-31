import { PcmResampler } from "./resampler.js";

class CaptureProcessor extends AudioWorkletProcessor {
  private readonly resampler: PcmResampler;
  constructor(options: AudioWorkletNodeOptions) {
    super();
    const raw: unknown = options.processorOptions;
    let outputRate: 16000 | 24000 = 16000;
    if (typeof raw === "object" && raw !== null && "outputRate" in raw) {
      if (raw.outputRate !== 16000 && raw.outputRate !== 24000) throw new Error("Invalid capture rate");
      outputRate = raw.outputRate;
    }
    this.resampler = new PcmResampler(sampleRate, outputRate);
  }
  process(inputs: Float32Array[][]): boolean {
    const channel = inputs[0]?.[0];
    if (channel !== undefined) {
      this.resampler.push(channel, buffer => this.port.postMessage(buffer, [buffer]));
    }
    return true;
  }
}
registerProcessor("capture", CaptureProcessor);
