import type { Measurement } from "../domain/session.js";
import { element } from "./dom.js";

export class Measurements {
  readonly element = element("section", "measurements");
  readonly summary = element("span", "rtf-summary", "RTF —");
  private readonly rtf = element("dd", "metric-value", "—");
  private readonly rtfLabel = element("dt", "metric-label", "Real-time factor");
  private readonly audio = element("dd", "metric-value", "—");
  private readonly latency = element("dd", "metric-value", "—");
  private readonly latencyLabel = element("dt", "metric-label", "Latency");
  private readonly queue = element("dd", "metric-value", "—");
  private readonly explanation = element("p", "field-hint", "Measurements appear after audio is processed.");

  constructor() {
    this.element.append(element("h2", "section-title", "Measurements"));
    const grid = element("dl", "metric-grid");
    for (const [label, value] of [
      [this.rtfLabel, this.rtf],
      [element("dt", "metric-label", "Audio duration"), this.audio],
      [this.latencyLabel, this.latency],
      [element("dt", "metric-label", "Input backlog"), this.queue],
    ]) {
      const cell = element("div", "metric");
      if (label !== undefined && value !== undefined) cell.append(label, value);
      grid.append(cell);
    }
    const guide = element("details", "measurement-guide");
    guide.append(element("summary", "", "About these measurements"), this.explanation,
      element("p", "field-hint", "RTF below 1 means faster than real time. Compare runs with the same inputs and settings."));
    this.element.append(grid, guide);
  }
  reset(): void {
    this.rtf.textContent = "—"; this.audio.textContent = "—";
    this.latency.textContent = "—"; this.queue.textContent = "—";
    this.rtfLabel.textContent = "Real-time factor";
    this.summary.textContent = "RTF —";
    this.latencyLabel.textContent = "Latency";
    this.explanation.textContent = "Measurements appear after audio is processed.";
  }
  update(value: Measurement): void {
    this.rtf.textContent = value.rtf === null ? "—" : `${value.rtf.toFixed(2)}×`;
    this.rtfLabel.textContent = value.basis === "input" ? "Input RTF" : "Output RTF";
    this.summary.textContent = `${this.rtfLabel.textContent} ${this.rtf.textContent}`;
    this.audio.textContent = `${value.audioSeconds.toFixed(2)} s`;
    this.latencyLabel.textContent = value.latencyLabel;
    this.latency.textContent = value.latencyMs === null ? "—" : `${Math.round(value.latencyMs)} ms`;
    this.queue.textContent = value.backlogSeconds === null ? "N/A" : `${value.backlogSeconds.toFixed(2)} s`;
    this.explanation.textContent = value.latencyLabel === "Compute to first PCM"
      ? "Input RTF = model and codec processing time ÷ microphone audio duration. Compute to first PCM sums processing time; it excludes live pacing, setup, network, and playback delay."
      : value.basis === "input"
      ? "Input RTF = model processing time ÷ microphone audio duration. Frame p95 is processing latency; network and playback delay are separate."
      : "Output RTF = generation time ÷ generated audio duration. First PCM measures server audio readiness, not audible playback latency. Model loading is excluded.";
  }
}
