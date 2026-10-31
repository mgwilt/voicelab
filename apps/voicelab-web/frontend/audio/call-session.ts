type AudioSessionType = "auto" | "playback" | "transient" | "transient-solo" | "ambient" | "play-and-record";
interface BrowserAudioSession { type: AudioSessionType }
export type CallAudioRouting = "play-and-record" | "unavailable" | "not-applied";

function isAudioSession(value: unknown): value is BrowserAudioSession {
  if (typeof value !== "object" || value === null || !("type" in value)) return false;
  return value.type === "auto" || value.type === "playback" || value.type === "transient"
    || value.type === "transient-solo" || value.type === "ambient" || value.type === "play-and-record";
}

/** Owns the page's communication audio mode until microphone teardown completes. */
export class CallAudioSession {
  private session: BrowserAudioSession | null = null;
  private previous: AudioSessionType = "auto";
  private failed = false;

  constructor(host: object = navigator) {
    try {
      if (!("audioSession" in host) || !isAudioSession(host.audioSession)) return;
      this.session = host.audioSession;
      this.previous = this.session.type;
      this.session.type = "play-and-record";
    } catch { this.failed = true; }
  }

  get routing(): CallAudioRouting {
    try {
      if (this.failed) return "not-applied";
      if (this.session === null) return "unavailable";
      return this.session.type === "play-and-record" ? "play-and-record" : "not-applied";
    } catch { return "not-applied"; }
  }

  close(): void {
    const session = this.session;
    this.session = null;
    try {
      // Do not overwrite a subsequent change made by another page feature.
      if (session?.type === "play-and-record") session.type = this.previous;
    } catch { /* An optional routing API must not prevent call cleanup. */ }
  }
}
