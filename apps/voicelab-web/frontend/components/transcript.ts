import { element } from "./dom.js";

export class Transcript {
  readonly element = element("section", "transcript");
  private readonly user = element("p", "transcript-text");
  private readonly assistant = element("p", "transcript-text");
  private readonly functionText = element("p", "transcript-text");
  private readonly empty = element("p", "transcript-empty field-hint", "Text from the conversation will appear here when the model provides it.");
  private readonly functions = element("div", "utterance");
  constructor() {
    this.element.append(element("h2", "section-title", "Conversation"));
    const user = element("div", "utterance");
    user.append(element("h3", "metric-label", "You"), this.user);
    const assistant = element("div", "utterance");
    assistant.append(element("h3", "metric-label", "Model"), this.assistant);
    this.functions.append(element("h3", "metric-label", "Function channel (not executed)"), this.functionText);
    this.functions.hidden = true;
    this.element.append(this.empty, user, assistant, this.functions);
    for (const text of [this.user, this.assistant, this.functionText]) text.dir = "auto";
  }
  update(speaker: "user" | "assistant" | "function", text: string): void {
    this.empty.hidden = true;
    if (speaker === "function") {
      this.functions.hidden = false;
      this.functionText.textContent = text;
    } else {
      (speaker === "user" ? this.user : this.assistant).textContent = text;
    }
  }
  reset(): void { this.empty.hidden = false; this.user.textContent = ""; this.assistant.textContent = ""; this.functionText.textContent = ""; this.functions.hidden = true; }
  show(visible: boolean): void { this.element.hidden = !visible; }
}
