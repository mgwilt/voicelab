import { element } from "../components/dom.js";
import type { Notice } from "./events.js";

/** Retains stage timings without mixing server delivery with audible playback. */
export class ConversationMetrics {
  readonly element = element("details", "conversation-metrics");
  private readonly rows = element("dl");
  private readonly values = new Map<string, string>();
  constructor() {
    this.element.append(element("summary", "", "Server timing details"), this.rows,
      element("p", "workflow-note", "First audio is measured at the server. Network and speaker playback add latency."));
  }
  update(event: Notice): string {
    const label = `${event.turn > 0 ? `Turn ${event.turn} · ` : ""}${event.stage.replaceAll("_", " ")}`;
    const parts: string[] = [];
    if (event.seconds !== null) parts.push(`${event.seconds.toFixed(2)} s`);
    if (event.outputRtf !== null) parts.push(`Output RTF ${event.outputRtf.toFixed(2)}`);
    if (event.inputRtf !== null) parts.push(`Input RTF ${event.inputRtf.toFixed(2)}`);
    if (event.text !== "") parts.push(event.text);
    this.values.set(label, parts.join(" · "));
    if (this.values.size > 60) { const oldest = this.values.keys().next().value; if (oldest !== undefined) this.values.delete(oldest); }
    this.rows.replaceChildren();
    for (const [name, value] of this.values) this.rows.append(element("dt", "", name), element("dd", "", value));
    return `${label} · ${parts.join(" · ")}`;
  }
  reset(): void { this.values.clear(); this.rows.replaceChildren(); }
}
