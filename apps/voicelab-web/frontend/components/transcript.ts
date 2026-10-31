import { element } from "./dom.js";

export class Transcript {
  readonly element = element("section", "transcript");
  private readonly user = element("p", "transcript-text");
  private readonly assistant = element("p", "transcript-text");
  constructor() {
    this.element.append(element("h2", "section-title", "Conversation"));
    const user = element("div", "utterance");
    user.append(element("h3", "metric-label", "You"), this.user);
    const assistant = element("div", "utterance");
    assistant.append(element("h3", "metric-label", "Model"), this.assistant);
    this.element.append(user, assistant);
  }
  update(speaker: "user" | "assistant", text: string): void {
    (speaker === "user" ? this.user : this.assistant).textContent = text;
  }
  reset(): void { this.user.textContent = ""; this.assistant.textContent = ""; }
  show(visible: boolean): void { this.element.hidden = !visible; }
}
