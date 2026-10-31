import { BrowserRecordingCapture, MAX_RECORDING_BYTES, recordingWav, type RecordingCapture } from "../audio/recording.js";
import { element, labeled } from "./dom.js";

interface Recording {
  readonly capture: RecordingCapture;
  readonly abort: AbortController;
  readonly chunks: Uint8Array[];
  bytes: number;
  timer: ReturnType<typeof setTimeout> | null;
}

/** Explicit record-or-upload input. Recorded audio never leaves the browser until a run. */
export class AudioInput {
  readonly element = element("div", "field");
  private readonly upload = element("input");
  private readonly record = element("button", "button-secondary", "Record audio");
  private readonly stop = element("button", "button-secondary", "Stop recording");
  private readonly clear = element("button", "button-secondary", "Clear audio");
  private readonly status = element("p", "field-hint", "No input audio selected.");
  private readonly listeners = new AbortController();
  private active: Recording | null = null;
  private recorded: File | null = null;
  private closing = false;
  private disabled = false;
  private disposed = false;

  constructor(label: string, private readonly capture: () => RecordingCapture = () => new BrowserRecordingCapture()) {
    this.upload.type = "file"; this.upload.accept = ".wav,audio/wav";
    for (const button of [this.record, this.stop, this.clear]) button.type = "button";
    this.status.setAttribute("role", "status"); this.status.setAttribute("aria-live", "polite");
    const actions = element("div", "actions"); actions.append(this.record, this.stop, this.clear);
    this.element.append(labeled(label, this.upload, "Upload mono 16 kHz PCM16 WAV, or record up to 30 seconds. Microphone access requires trusted HTTPS."), actions, this.status);
    const options = { signal: this.listeners.signal };
    this.record.addEventListener("click", () => { void this.start(); }, options);
    this.stop.addEventListener("click", () => { if (this.active !== null) void this.finish(this.active, true); }, options);
    this.clear.addEventListener("click", () => {
      this.recorded = null; this.upload.value = ""; this.status.textContent = "No input audio selected."; this.render();
    }, options);
    this.upload.addEventListener("change", () => {
      this.recorded = null;
      this.status.textContent = this.upload.files?.item(0)?.name ?? "No input audio selected.";
      this.render();
    }, options);
    this.render();
  }

  read(): File | null {
    if (this.active !== null || this.closing) throw new Error("Stop recording before starting the test.");
    const file = this.recorded ?? this.upload.files?.item(0) ?? null;
    if (file !== null && (file.size === 0 || file.size > 1025536)) throw new Error("Choose a mono 16 kHz PCM16 WAV of up to 30 seconds.");
    return file;
  }

  private async start(): Promise<void> {
    if (this.disabled || this.disposed || this.active !== null || this.closing) return;
    const run: Recording = { capture: this.capture(), abort: new AbortController(), chunks: [], bytes: 0, timer: null };
    this.active = run; this.recorded = null; this.upload.value = "";
    this.status.textContent = "Waiting for microphone permission…"; this.render();
    try {
      await run.capture.open(run.abort.signal, pcm => this.receive(run, pcm));
      if (this.active !== run || run.abort.signal.aborted) return;
      this.status.textContent = "Recording. Stops automatically after 30 seconds.";
      run.timer = setTimeout(() => { void this.finish(run, true); }, 30_000);
    } catch (error) {
      if (this.active !== run) return;
      this.status.textContent = error instanceof Error ? error.message : "Microphone could not start.";
      await this.finish(run, false);
    }
  }

  private receive(run: Recording, pcm: ArrayBuffer): void {
    if (this.active !== run || run.abort.signal.aborted) return;
    if (pcm.byteLength % 2 !== 0) {
      this.status.textContent = "Microphone returned invalid PCM16 audio."; void this.finish(run, false); return;
    }
    const count = Math.min(pcm.byteLength, MAX_RECORDING_BYTES - run.bytes);
    if (count > 0) { run.chunks.push(new Uint8Array(pcm.slice(0, count))); run.bytes += count; }
    if (run.bytes === MAX_RECORDING_BYTES) void this.finish(run, true);
  }

  private async finish(run: Recording, save: boolean): Promise<void> {
    if (this.active !== run) return;
    this.active = null; this.closing = true; run.abort.abort();
    if (run.timer !== null) clearTimeout(run.timer);
    if (save && run.bytes > 0) {
      this.recorded = recordingWav(run.chunks);
      this.status.textContent = `Recorded ${(run.bytes / 32000).toFixed(1)} seconds. Ready to send.`;
    } else if (save) this.status.textContent = "No audio captured. Record for at least a moment.";
    run.chunks.length = 0; this.render();
    try { await run.capture.close(); }
    catch { if (!this.disposed) this.status.textContent = "The microphone could not close cleanly. Reload before recording again."; }
    finally { this.closing = false; this.render(); }
  }

  private render(): void {
    const busy = this.active !== null || this.closing;
    this.upload.disabled = this.disabled || busy;
    this.record.disabled = this.disabled || busy;
    this.stop.disabled = this.disabled || this.active === null;
    this.clear.disabled = this.disabled || busy;
  }
  setDisabled(disabled: boolean): void {
    this.disabled = disabled;
    if (disabled && this.active !== null) {
      this.status.textContent = "Recording cancelled.";
      void this.finish(this.active, false);
    }
    this.render();
  }
  dispose(): void {
    this.disposed = true; this.listeners.abort();
    if (this.active !== null) void this.finish(this.active, false);
    this.recorded = null; this.element.remove();
  }
}
