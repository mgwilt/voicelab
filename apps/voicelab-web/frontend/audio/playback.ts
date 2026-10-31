/** One audio context and bounded playback queue per session. */
export class Playback {
  private contextValue: AudioContext | null = null;
  private readonly sources = new Set<AudioBufferSourceNode>();
  private nextPlay = 0;
  private readonly drained = new Set<() => void>();

  get context(): AudioContext {
    if (this.contextValue === null) throw new Error("Audio context is not open");
    return this.contextValue;
  }
  async open(signal: AbortSignal): Promise<void> {
    signal.throwIfAborted();
    this.contextValue = new AudioContext();
    await this.contextValue.resume();
    signal.throwIfAborted();
  }
  play(pcm: ArrayBuffer, rate: number, maxQueueSeconds: number): void {
    if (pcm.byteLength === 0 || pcm.byteLength % 2 !== 0) throw new Error("Invalid PCM16 audio");
    const context = this.context;
    const buffer = context.createBuffer(1, pcm.byteLength / 2, rate);
    const channel = buffer.getChannelData(0);
    const view = new DataView(pcm);
    for (let i = 0; i < channel.length; i++) channel[i] = view.getInt16(i * 2, true) / 32768;
    this.nextPlay = Math.max(this.nextPlay, context.currentTime + 0.04);
    if (this.nextPlay + buffer.duration - context.currentTime > maxQueueSeconds) {
      throw new Error("Playback queue exceeded its time limit");
    }
    const source = context.createBufferSource();
    source.buffer = buffer;
    source.connect(context.destination);
    this.sources.add(source);
    source.onended = () => {
      this.sources.delete(source);
      source.disconnect();
      if (this.sources.size === 0) this.drained.forEach(done => done());
    };
    source.start(this.nextPlay);
    this.nextPlay += buffer.duration;
  }
  async drain(signal: AbortSignal): Promise<void> {
    signal.throwIfAborted();
    if (this.sources.size === 0) return;
    await new Promise<void>((resolve, reject) => {
      const finish = (): void => {
        signal.removeEventListener("abort", abort);
        this.drained.delete(finish);
        clearTimeout(timer);
        resolve();
      };
      const abort = (): void => {
        this.drained.delete(finish);
        clearTimeout(timer);
        signal.removeEventListener("abort", abort);
        reject(new DOMException("Playback cancelled", "AbortError"));
      };
      const timer = setTimeout(abort, 130_000);
      this.drained.add(finish);
      signal.addEventListener("abort", abort, { once: true });
    });
  }
  async close(): Promise<void> {
    for (const source of this.sources) {
      source.onended = null;
      source.stop();
      source.disconnect();
    }
    this.sources.clear();
    this.drained.forEach(done => done());
    this.drained.clear();
    const context = this.contextValue;
    this.contextValue = null;
    if (context !== null && context.state !== "closed") await context.close();
  }
}
