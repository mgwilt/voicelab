import { element, labeled } from "../../components/dom.js";
import { ReferenceAudioInput, type ReferenceAudio } from "../synthesis/reference.js";

export interface HistoryTurn { readonly role: "user" | "assistant"; readonly content: string; readonly audio: ReferenceAudio | null; }
interface Exchange { readonly node: HTMLElement; readonly user: HTMLTextAreaElement; readonly assistant: HTMLTextAreaElement; readonly remove: HTMLButtonElement; readonly audio: ReferenceAudioInput; }

/** Explicit prior exchanges, including the original audio of previous user turns. */
export class HistoryInput {
  readonly element = element("div");
  private readonly add = element("button", "button-secondary", "Add previous exchange");
  private readonly exchanges: Exchange[] = [];
  private disabled = false;
  constructor() {
    this.add.type = "button";
    this.add.addEventListener("click", () => this.append());
    this.element.append(this.add);
  }
  private append(): void {
    if (this.disabled || this.exchanges.length >= 10) return;
    const node = element("div");
    const user = element("textarea"); const assistant = element("textarea");
    user.maxLength = assistant.maxLength = 4000;
    user.rows = assistant.rows = 2;
    const remove = element("button", "button-secondary", "Remove exchange"); remove.type = "button";
    const audio = new ReferenceAudioInput("Previous user audio (optional)");
    const exchange: Exchange = { node, user, assistant, remove, audio };
    remove.addEventListener("click", () => {
      const index = this.exchanges.indexOf(exchange);
      if (index !== -1) this.exchanges.splice(index, 1);
      node.remove(); this.setDisabled(this.disabled);
    });
    node.append(labeled("Previous user message", user), audio.element, labeled("Previous assistant answer", assistant), remove);
    this.exchanges.push(exchange); this.element.insertBefore(node, this.add);
    this.setDisabled(this.disabled);
  }
  async read(): Promise<readonly HistoryTurn[]> {
    const turns: HistoryTurn[] = [];
    for (const exchange of this.exchanges) {
      const audio = await exchange.audio.read();
      if ((!exchange.user.value.trim() && audio === null) || !exchange.assistant.value.trim()) throw new Error("Complete both sides of each previous exchange.");
      turns.push({ role: "user", content: exchange.user.value, audio }, { role: "assistant", content: exchange.assistant.value, audio: null });
    }
    if (turns.reduce((sum, turn) => sum + turn.content.length, 0) > 20000) throw new Error("Previous messages exceed 20000 characters.");
    return turns;
  }
  setDisabled(value: boolean): void {
    this.disabled = value; this.add.disabled = value || this.exchanges.length >= 10;
    for (const exchange of this.exchanges) {
      exchange.user.disabled = exchange.assistant.disabled = exchange.remove.disabled = value;
      exchange.audio.setDisabled(value);
    }
  }
}
