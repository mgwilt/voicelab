import { element, integerInput, labeled, readInteger } from "../../components/dom.js";
import type { ControlPanel } from "../../components/panel.js";
import type { DuplexModel, Emit, ExperimentSession } from "../../domain/experiment.js";
import { DuplexSession } from "./session.js";

export class DuplexPanel implements ControlPanel {
  readonly element = element("div", "control-panel");
  private readonly prompt = element("textarea");
  private readonly seed = integerInput(0, 0, 2147483647);

  constructor(private readonly model: DuplexModel) {
    this.prompt.value = "Be concise and answer in one sentence.";
    this.prompt.rows = 3; this.prompt.maxLength = 2000;
    this.element.append(labeled("Instructions", this.prompt, "Use an ASCII prompt for this model."),
      labeled("Seed", this.seed, "Repeat a seed to hold sampling settings constant."),
      element("p", "workflow-note", "Your microphone stays live while the model speaks. Headphones reduce feedback. Sessions last up to two minutes."));
  }
  createSession(emit: Emit): ExperimentSession {
    if ([...this.prompt.value].some(character => character.codePointAt(0) === undefined || (character.codePointAt(0) ?? 0) > 127)) {
      throw new Error("This model requires an ASCII prompt.");
    }
    return new DuplexSession(this.model, { type: "start", prompt: this.prompt.value, seed: readInteger(this.seed) }, emit);
  }
  setDisabled(disabled: boolean): void { this.prompt.disabled = disabled; this.seed.disabled = disabled; }
  dispose(): void { this.element.remove(); }
}
