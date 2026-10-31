import { element, integerInput, labeled, readInteger } from "../../components/dom.js";
import type { ControlPanel } from "../../components/panel.js";
import type { BreezeModel, Emit, ExperimentSession } from "../../domain/experiment.js";
import { SynthesisSession } from "../synthesis/session.js";

export class BreezePanel implements ControlPanel {
  readonly element = element("div", "control-panel");
  private readonly text = element("textarea");
  private readonly instruction = element("textarea");
  private readonly seed = integerInput(0, 0, 4294967295);
  private readonly frames = integerInput(750, 1, 750);
  private readonly guidance = element("input");

  constructor(private readonly model: BreezeModel) {
    this.text.value = "Hello. This is a speech test in Voice Lab.";
    this.text.rows = 4; this.text.maxLength = 4000; this.text.required = true;
    this.instruction.value = "A clear, natural conversational voice.";
    this.instruction.rows = 2; this.instruction.maxLength = 2000; this.instruction.required = true;
    this.guidance.type = "number"; this.guidance.min = "0.1"; this.guidance.max = "20";
    this.guidance.step = "0.1"; this.guidance.value = "4"; this.guidance.required = true;
    const row = element("div", "field-grid");
    row.append(labeled("Seed", this.seed), labeled("Guidance", this.guidance), labeled("Frame limit", this.frames));
    this.element.append(labeled("Text to speak", this.text), labeled("Voice instruction", this.instruction,
      "Describe the voice, delivery, or emotion."), row,
      element("p", "workflow-note", "Guidance controls how strongly the voice follows the instruction. A frame-limit cutoff is reported as incomplete."));
  }
  createSession(emit: Emit): ExperimentSession {
    if (this.text.value.trim() === "" || this.instruction.value.trim() === "") {
      throw new Error("Enter text and a voice instruction.");
    }
    const guidance = this.guidance.valueAsNumber;
    if (!this.guidance.checkValidity() || !Number.isFinite(guidance) || guidance <= 0) {
      this.guidance.reportValidity();
      throw new Error("Guidance must be greater than 0 and at most 20.");
    }
    return new SynthesisSession({ model: this.model, request: {
      text: this.text.value, instruction: this.instruction.value,
      seed: readInteger(this.seed), cfg_scale: guidance, max_frames: readInteger(this.frames),
    } }, emit);
  }
  setDisabled(disabled: boolean): void {
    for (const input of [this.text, this.instruction, this.seed, this.frames, this.guidance]) input.disabled = disabled;
  }
  dispose(): void { this.element.remove(); }
}
