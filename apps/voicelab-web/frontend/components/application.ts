import { ConversationView } from "../conversation/view.js";
import type { Model } from "../domain/session.js";
import { element } from "./dom.js";
import { Lab } from "./lab.js";
import type { ControlPanel } from "./panel.js";

/** Navigation owns which workflow may hold the microphone and inference lease. */
export class Application {
  readonly element = element("div", "application");
  private readonly conversation: ConversationView;
  private readonly direct: Lab;
  private readonly navigation = element("nav", "workflow-tabs");
  private readonly conversationTab = element("button", "", "Conversation");
  private readonly directTab = element("button", "", "Direct run");
  private readonly lifetime = new AbortController();
  private switching = false;

  constructor(models: readonly Model[], createPanel: (model: Model) => ControlPanel) {
    this.conversation = new ConversationView(models, createPanel);
    this.direct = new Lab(models, createPanel);
    this.direct.element.hidden = true;
    this.navigation.setAttribute("aria-label", "Voice workflow");
    this.conversationTab.setAttribute("aria-current", "page");
    this.navigation.append(this.conversationTab, this.directTab);
    this.element.append(this.navigation, this.conversation.element, this.direct.element);
    this.conversationTab.addEventListener("click", () => { void this.show(true); }, { signal: this.lifetime.signal });
    this.directTab.addEventListener("click", () => { void this.show(false); }, { signal: this.lifetime.signal });
  }
  private async show(conversation: boolean): Promise<void> {
    if (this.switching || this.lifetime.signal.aborted) return;
    this.switching = true;
    this.conversationTab.disabled = this.directTab.disabled = true;
    try {
      await this.conversation.stop(); await this.direct.cancel();
      if (this.lifetime.signal.aborted) return;
      this.conversation.element.hidden = !conversation; this.direct.element.hidden = conversation;
      this.conversationTab.setAttribute("aria-current", conversation ? "page" : "false");
      this.directTab.setAttribute("aria-current", conversation ? "false" : "page");
    } finally { this.switching = false; this.conversationTab.disabled = this.directTab.disabled = false; }
  }
  async dispose(): Promise<void> {
    this.lifetime.abort(); await this.conversation.dispose(); await this.direct.dispose();
  }
}
