import type { CallAudioRouting } from "./call-session.js";

/** Configure local capture; never infer effective processing from requested constraints. */
export interface MicrophoneProcessingReport {
  readonly callAudioRouting: CallAudioRouting;
  readonly echoCancellation: boolean | null;
  readonly noiseSuppression: boolean | null;
  readonly autoGainControl: boolean | null;
  readonly configurationFailed: boolean;
}

export type ReportMicrophoneProcessing = (report: MicrophoneProcessingReport | null) => void;

export class MicrophoneProcessing {
  private current: MediaStreamTrack | null = null;
  private pending: Promise<void> | null = null;
  private failed = false;
  private closed = false;

  constructor(private readonly report: ReportMicrophoneProcessing, private readonly routing: () => CallAudioRouting = () => "unavailable") {}

  configure(track: MediaStreamTrack): Promise<void> {
    if (this.closed || track.kind !== "audio" || track.readyState !== "live") return Promise.resolve();
    if (this.current === track && this.pending !== null) return this.pending;
    this.current = track;
    this.failed = false;
    this.pending = this.apply(track);
    return this.pending;
  }

  private async apply(track: MediaStreamTrack): Promise<void> {
    let failed = false;
    try {
      await track.applyConstraints({
        ...track.getConstraints(),
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      });
    } catch {
      // A browser can reject processing changes while retaining a usable microphone.
      failed = true;
    }
    if (this.closed || this.current !== track || track.readyState !== "live") return;
    this.failed = failed;
    this.verify();
  }

  verify(): void {
    if (this.closed || this.current === null || this.current.readyState !== "live") return;
    let settings: MediaTrackSettings = {};
    try { settings = this.current.getSettings(); } catch { /* Unreported is distinct from off. */ }
    this.report({
      callAudioRouting: this.routing(),
      echoCancellation: typeof settings.echoCancellation === "boolean" ? settings.echoCancellation : null,
      noiseSuppression: settings.noiseSuppression ?? null,
      autoGainControl: settings.autoGainControl ?? null,
      configurationFailed: this.failed,
    });
  }

  close(): void {
    this.closed = true;
    this.current = null;
    this.pending = null;
  }
}
