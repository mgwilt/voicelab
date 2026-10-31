import { Microphone } from "./microphone.js";

export const RECORDING_RATE = 16000;
export const MAX_RECORDING_BYTES = RECORDING_RATE * 2 * 30;

export interface RecordingCapture {
  open(signal: AbortSignal, receive: (pcm: ArrayBuffer) => void): Promise<void>;
  close(): Promise<void>;
}

/** Owns device permission, the capture worklet, and its audio context. */
export class BrowserRecordingCapture implements RecordingCapture {
  private readonly microphone = new Microphone();
  private context: AudioContext | null = null;

  async open(signal: AbortSignal, receive: (pcm: ArrayBuffer) => void): Promise<void> {
    signal.throwIfAborted();
    const context = new AudioContext();
    this.context = context;
    await context.resume();
    signal.throwIfAborted();
    await this.microphone.open(context, signal);
    signal.throwIfAborted();
    this.microphone.start(context, receive, RECORDING_RATE);
  }

  async close(): Promise<void> {
    this.microphone.close();
    const context = this.context;
    this.context = null;
    if (context !== null && context.state !== "closed") await context.close();
  }
}

/** Capture supplies little-endian mono PCM16; write a standard RIFF container. */
export function recordingWav(chunks: readonly Uint8Array[]): File {
  const size = chunks.reduce((total, chunk) => total + chunk.byteLength, 0);
  if (size === 0 || size > MAX_RECORDING_BYTES || chunks.some(chunk => chunk.byteLength % 2 !== 0)) {
    throw new Error("A recording must contain up to 30 seconds of mono PCM16 audio.");
  }
  const buffer = new ArrayBuffer(44 + size);
  const view = new DataView(buffer);
  const bytes = new Uint8Array(buffer);
  const text = (offset: number, value: string): void => {
    for (let i = 0; i < value.length; i++) view.setUint8(offset + i, value.charCodeAt(i));
  };
  text(0, "RIFF"); view.setUint32(4, 36 + size, true); text(8, "WAVE");
  text(12, "fmt "); view.setUint32(16, 16, true); view.setUint16(20, 1, true);
  view.setUint16(22, 1, true); view.setUint32(24, RECORDING_RATE, true);
  view.setUint32(28, RECORDING_RATE * 2, true); view.setUint16(32, 2, true);
  view.setUint16(34, 16, true); text(36, "data"); view.setUint32(40, size, true);
  let offset = 44;
  for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.byteLength; }
  return new File([buffer], "recording.wav", { type: "audio/wav" });
}
