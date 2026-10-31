import { PcmResampler } from "./resampler.js";

class CaptureProcessor extends AudioWorkletProcessor {
  private readonly resampler = new PcmResampler(sampleRate);
  process(inputs: Float32Array[][]): boolean {
    const channel = inputs[0]?.[0];
    if (channel !== undefined) {
      this.resampler.push(channel, buffer => this.port.postMessage(buffer, [buffer]));
    }
    return true;
  }
}
registerProcessor("capture", CaptureProcessor);
