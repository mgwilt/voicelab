import { element, integerInput, labeled, readInteger } from "../../components/dom.js";
import type { ControlPanel } from "../../components/panel.js";
import type { Emit, ExperimentSession, SesameModel } from "../../domain/experiment.js";
import { SynthesisSession } from "../synthesis/session.js";

export class SynthesisPanel implements ControlPanel {
  readonly element = element("div", "control-panel");
  private readonly text = element("textarea");
  private readonly speaker = element("select");
  private readonly seed = integerInput(42, 0, 4294967295);
  private readonly frames = integerInput(250, 4, 1500);
  constructor(private readonly model: SesameModel) {
    this.text.value = "Hello. This is a speech test in Voice Lab.";
    this.text.rows = 5; this.text.maxLength = 8000; this.text.required = true;
    for (const value of ["0", "1"]) {
      const option = element("option", "", value);
      option.value = value; this.speaker.append(option);
    }
    const row = element("div", "field-grid");
    row.append(labeled("Speaker", this.speaker), labeled("Seed", this.seed), labeled("Frame limit", this.frames));
    this.element.append(labeled("Text to speak", this.text), row,
      element("p", "workflow-note", "Each frame represents 80 ms. Hitting the frame limit marks the output incomplete. Speaker IDs are conversation roles, not preset voices."));
  }
  createSession(emit: Emit): ExperimentSession {
    if (this.text.value.trim() === "") throw new Error("Enter text to speak.");
    const speaker = this.speaker.value;
    if (speaker !== "0" && speaker !== "1") throw new Error("Invalid speaker");
    return new SynthesisSession({ model: this.model, request: {
      text: this.text.value, speaker: speaker === "0" ? 0 : 1,
      seed: readInteger(this.seed), max_frames: readInteger(this.frames),
    } }, emit);
  }
  setDisabled(disabled: boolean): void {
    for (const input of [this.text, this.speaker, this.seed, this.frames]) input.disabled = disabled;
  }
  dispose(): void { this.element.remove(); }
}
