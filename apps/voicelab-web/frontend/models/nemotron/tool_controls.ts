import { element, integerInput, labeled, readInteger } from "../../components/dom.js";
import { AudioInput } from "../../components/audio-input.js";
import type { ToolQuestion } from "./tool_session.js";

export class ToolControls {
  readonly element = element("div");
  private readonly audio = new AudioInput("Spoken tool request");
  private readonly tools = element("textarea");
  private readonly response = element("textarea");
  private readonly silence = integerInput(16, 0, 30);
  constructor() {
    this.tools.rows = 6; this.tools.maxLength = 12000;
    this.tools.value = JSON.stringify([{ type: "function", function: { name: "generate_random_number",
      description: "Generate a random number between the requested minimum and maximum.",
      parameters: { type: "object", properties: { minimum: { type: "integer" }, maximum: { type: "integer" } }, required: ["minimum", "maximum"] } } }], null, 2);
    this.response.rows = 4; this.response.maxLength = 4000;
    this.response.value = JSON.stringify({ tool_name: "generate_random_number", response: { result: 73 } }, null, 2);
    this.element.append(this.audio.element,
      labeled("Tool definitions (JSON)", this.tools), labeled("Supplied response (JSON)", this.response),
      labeled("Trailing silence (seconds)", this.silence),
      element("p", "workflow-note", "Runs NVIDIA's two-pass offline workflow. No tools are executed. The model must request the supplied tool; playback includes the inserted tool-response interval. RTF counts both passes."));
  }
  read(prompt: string, seed: number): ToolQuestion {
    const file = this.audio.read();
    if (file === null || file.size === 0 || file.size > 1025536) throw new Error("Choose a mono 16 kHz PCM16 WAV of up to 30 seconds.");
    const tools: unknown = JSON.parse(this.tools.value);
    const response: unknown = JSON.parse(this.response.value);
    if (!Array.isArray(tools) || tools.length === 0 || tools.length > 8) throw new Error("Supply an array of one to eight tool definitions.");
    if (typeof response !== "object" || response === null || Array.isArray(response)) throw new Error("Supply a response object with tool_name and response.");
    return { file, prompt, seed, trailingSilence: readInteger(this.silence), tools, response };
  }
  setDisabled(disabled: boolean): void {
    this.audio.setDisabled(disabled); this.tools.disabled = disabled; this.response.disabled = disabled; this.silence.disabled = disabled;
  }
  dispose(): void { this.audio.dispose(); this.element.remove(); }
}
