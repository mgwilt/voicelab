import { element } from "../components/dom.js";
import type { MicrophoneProcessingReport } from "../audio/processing.js";

const state = (enabled: boolean | null): string => enabled === null ? "Not reported" : enabled ? "On" : "Off";

/** Remains visible when call status changes from listening to speaking. */
export class MicrophoneStatus {
  readonly element = element("section", "microphone-status");
  private readonly summary = element("p", "workflow-note");
  private readonly values = element("p", "workflow-note");

  constructor() {
    this.summary.setAttribute("role", "status");
    const details = element("details");
    details.append(element("summary", "", "Microphone processing"), this.values);
    this.element.append(this.summary, details);
    this.update(null);
  }

  update(report: MicrophoneProcessingReport | null): void {
    this.element.hidden = report === null;
    if (report === null) { this.summary.textContent = this.values.textContent = ""; return; }
    this.summary.textContent = report.echoCancellation === true
      ? "Browser reports echo cancellation on. Speaker echo can still reach the microphone."
      : report.echoCancellation === false
        ? "Microphone echo cancellation is off. Use headphones to prevent the assistant hearing itself."
        : "This browser did not report echo cancellation. Use headphones if the assistant hears itself.";
    if (report.configurationFailed) this.summary.textContent += " The browser could not apply the requested microphone processing.";
    const routing = report.callAudioRouting === "play-and-record" ? "Play and record" : report.callAudioRouting === "unavailable" ? "Browser managed (call routing API unavailable)" : "Could not apply call audio routing";
    this.values.textContent = `Call audio routing: ${routing}. Echo cancellation: ${state(report.echoCancellation)}. Noise suppression: ${state(report.noiseSuppression)}. Automatic gain: ${state(report.autoGainControl)}. Reported by the browser; this does not measure echo removal.`;
  }
}
