import { element, integerInput, labeled, readInteger } from "../../components/dom.js";
import type { ControlPanel } from "../../components/panel.js";
import type { DuplexModel, Emit, VoiceSession } from "../../domain/session.js";
import { ToolControls } from "./tool_controls.js";
import { ToolSession } from "./tool_session.js";
import { DuplexSession } from "./session.js";

export class DuplexPanel implements ControlPanel {
  readonly element = element("div", "control-panel");
  private readonly prompt = element("textarea");
  private readonly mode = element("select");
  private readonly events = new AbortController();
  private disabled = false;
  private readonly tools = new ToolControls();
  private readonly liveNote = element("p", "workflow-note", "Your microphone stays live while the model speaks. Headphones reduce feedback. Sessions last up to two minutes.");
  private readonly seed = integerInput(0, 0, 2147483647);

  constructor(private readonly model: DuplexModel) {
    for (const [value, title] of [["live", "Live conversation"], ["tools", "Offline tool-response test"]] as const) {
      const option = element("option", "", title); option.value = value; this.mode.append(option);
    }
    this.updateMode();
    this.mode.addEventListener("change", () => this.updateMode(), { signal: this.events.signal });
    this.element.append(labeled("Workflow", this.mode));
    this.prompt.value = "Be concise and answer in one sentence.";
    this.prompt.rows = 3; this.prompt.maxLength = 2000;
    this.element.append(labeled("Instructions", this.prompt, "Use an ASCII prompt for this model."),
      labeled("Seed", this.seed, "Repeat a seed to hold sampling settings constant."),
      this.liveNote, this.tools.element);
  }
  async conversationConfig(): Promise<object> { return { type: "start", prompt: this.prompt.value, seed: readInteger(this.seed) }; }
  prepareConversation(): void { this.mode.hidden = true; this.tools.element.hidden = true; }
  createSession(emit: Emit): VoiceSession {
    if ([...this.prompt.value].some(character => character.codePointAt(0) === undefined || (character.codePointAt(0) ?? 0) > 127)) {
      throw new Error("This model requires an ASCII prompt.");
    }
    if (this.mode.value === "tools") return new ToolSession(this.tools.read(this.prompt.value, readInteger(this.seed)), emit);
    return new DuplexSession(this.model, { type: "start", prompt: this.prompt.value, seed: readInteger(this.seed) }, emit);
  }
  private updateMode(): void {
    const toolMode = this.mode.value === "tools";
    this.tools.element.hidden = !toolMode;
    this.liveNote.hidden = toolMode;
    this.tools.setDisabled(this.disabled || !toolMode);
  }
  setDisabled(disabled: boolean): void {
    this.disabled = disabled;
    this.prompt.disabled = disabled; this.seed.disabled = disabled; this.mode.disabled = disabled;
    this.updateMode();
  }
  dispose(): void { this.events.abort(); this.tools.dispose(); this.element.remove(); }
}
