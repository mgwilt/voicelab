import { acquireStream } from "./permission.js";

export class Microphone {
  private stream: MediaStream | null = null;
  private capture: AudioWorkletNode | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private filter: BiquadFilterNode | null = null;

  async open(context: AudioContext, signal: AbortSignal): Promise<void> {
    if (!window.isSecureContext || navigator.mediaDevices === undefined) {
      throw new Error("Microphone access requires trusted HTTPS on this device.");
    }
    signal.throwIfAborted();
    this.stream = await acquireStream(navigator.mediaDevices.getUserMedia({ audio: {
      channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true,
    } }), signal);
    await context.audioWorklet.addModule("/static/dist/capture-worklet.js");
    signal.throwIfAborted();
  }
  start(context: AudioContext, send: (pcm: ArrayBuffer) => void): void {
    if (this.stream === null || this.capture !== null) throw new Error("Invalid capture state");
    this.source = context.createMediaStreamSource(this.stream);
    this.filter = context.createBiquadFilter();
    this.filter.type = "lowpass";
    this.filter.frequency.value = 7200;
    this.capture = new AudioWorkletNode(context, "capture");
    this.capture.port.onmessage = (event: MessageEvent<unknown>) => {
      if (event.data instanceof ArrayBuffer) send(event.data);
    };
    this.source.connect(this.filter).connect(this.capture).connect(context.destination);
  }
  close(): void {
    if (this.capture !== null) {
      this.capture.port.onmessage = null;
      this.capture.port.close();
      this.capture.disconnect();
    }
    this.source?.disconnect();
    this.filter?.disconnect();
    this.stream?.getTracks().forEach(track => track.stop());
    this.capture = null; this.source = null; this.filter = null; this.stream = null;
  }
}
