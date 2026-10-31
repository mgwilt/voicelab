import { element, labeled } from "../components/dom.js";
import type { ControlPanel } from "../components/panel.js";
import type { Model } from "../domain/session.js";
import { ConversationClient, type ConversationConnection, type ConnectionFactory } from "./client.js";
import { ConversationMetrics } from "./metrics.js";
import type { Notice } from "./events.js";

export class ConversationView {
  readonly element = element("section", "conversation");
  private readonly picker = element("select");
  private readonly mode = element("select");
  private readonly settings = element("details", "conversation-settings");
  private readonly panelHost = element("div");
  private readonly prompt = element("textarea");
  private readonly feed = element("div", "conversation-feed");
  private readonly status = element("p", "session-status", "Choose a voice, then start a conversation.");
  private readonly metrics = new ConversationMetrics();
  private readonly timing = element("p", "conversation-timing");
  private readonly start = element("button", "button-primary", "Start conversation");
  private readonly end = element("button", "button-secondary", "End call");
  private readonly mute = element("button", "button-secondary", "Mute");
  private readonly interrupt = element("button", "button-secondary", "Interrupt");
  private readonly media = element("div", "conversation-media");
  private readonly lifetime = new AbortController();
  private readonly messages = new Map<string, HTMLElement>();
  private client: ConversationConnection | null = null;
  private panel: ControlPanel | null = null;
  private selected: Model | null = null;
  private muted = false;
  private pending = false;
  private generation = 0;

  constructor(private readonly models: readonly Model[], private readonly createPanel: (model: Model) => ControlPanel, private readonly connection: ConnectionFactory = (event, status, ended) => new ConversationClient(event, status, ended)) {
    for (const model of models) {
      const option = element("option", "", model.label); option.value = model.id; option.disabled = !model.available || (model.conversationModes?.length ?? 0) === 0;
      this.picker.append(option);
    }
    this.prompt.value = "Answer briefly and naturally. Use plain spoken language.";
    this.prompt.maxLength = 2000; this.prompt.rows = 3;
    this.settings.append(element("summary", "", "Voice and conversation settings"), labeled("Conversation mode", this.mode), labeled("Assistant instructions (cascade)", this.prompt), this.panelHost);
    this.feed.setAttribute("role", "log"); this.feed.setAttribute("aria-live", "polite");
    this.status.setAttribute("role", "status");
    const actions = element("div", "conversation-actions"); actions.append(this.start, this.mute, this.interrupt, this.end);
    const footer = element("div", "conversation-footer"); footer.append(this.status, this.timing, this.media, actions);
    this.element.append(labeled("Voice model", this.picker), this.settings, this.feed, this.metrics.element, footer);
    this.end.disabled = this.mute.disabled = this.interrupt.disabled = true;
    const signal = this.lifetime.signal;
    this.picker.addEventListener("change", () => { void this.select(); }, { signal });
    this.mode.addEventListener("change", () => this.updateControls(), { signal });
    this.start.addEventListener("click", () => { void this.begin(); }, { signal });
    this.end.addEventListener("click", () => { void this.stop(); }, { signal });
    this.mute.addEventListener("click", () => { this.muted = !this.muted; this.client?.mute(this.muted); this.mute.textContent = this.muted ? "Unmute" : "Mute"; this.mute.setAttribute("aria-pressed", String(this.muted)); }, { signal });
    this.interrupt.addEventListener("click", () => { void this.client?.interrupt().catch(error => this.showError(error)); }, { signal });
    const first = models.find(model => model.available && model.id === "kokoro" && (model.conversationModes?.length ?? 0) > 0) ?? models.find(model => model.available && (model.conversationModes?.length ?? 0) > 0);
    if (first !== undefined) { this.picker.value = first.id; void this.select(); }
    else { this.start.disabled = true; this.status.textContent = "No conversation runtimes are available. Open Direct run for other workflows."; }
  }
  private async select(): Promise<void> {
    const id = this.picker.value;
    await this.stop();
    if (this.lifetime.signal.aborted || this.picker.value !== id) return;
    this.selected = this.models.find(model => model.id === id) ?? null;
    this.panel?.dispose(); this.panel = null; this.panelHost.replaceChildren(); this.mode.replaceChildren();
    const model = this.selected;
    if (model === null) return;
    this.panel = this.createPanel(model); this.panel.prepareConversation?.(); this.panelHost.append(this.panel.element);
    const modes = model.conversationModes ?? [];
    for (const mode of modes) { const option = element("option", "", { cascade: "Local voice cascade", turn: "Turn-based speech", duplex: "Live duplex" }[mode]); option.value = mode; this.mode.append(option); }
    this.mode.value = modes[0] ?? "cascade";
    this.feed.replaceChildren(); this.messages.clear(); this.timing.textContent = ""; this.metrics.reset();
    this.updateControls();
  }
  private updateControls(): void {
    const busy = this.client !== null || this.pending;
    this.start.disabled = busy || this.selected === null;
    this.end.disabled = !busy; this.mute.disabled = !busy;
    this.interrupt.disabled = !busy || this.mode.value === "duplex";
    this.mode.disabled = busy; this.prompt.disabled = busy;
    this.prompt.parentElement?.toggleAttribute("hidden", this.mode.value !== "cascade");
    const mode = this.mode.value;
    if (mode === "cascade" || mode === "turn" || mode === "duplex") this.panel?.prepareConversation?.(mode);
    this.panel?.setDisabled(busy);
  }
  private async begin(): Promise<void> {
    if (this.pending || this.client !== null || this.panel?.conversationConfig === undefined || this.selected === null) return;
    const generation = ++this.generation;
    this.pending = true; this.updateControls();
    try {
      const settings = await this.panel.conversationConfig();
      if (generation !== this.generation || !this.pending || this.lifetime.signal.aborted) return;
      const mode = this.mode.value;
      if (mode !== "cascade" && mode !== "turn" && mode !== "duplex") throw new Error("Invalid conversation mode");
      this.feed.replaceChildren(); this.messages.clear(); this.timing.textContent = ""; this.metrics.reset();
      const client = this.connection(event => this.receive(event), message => { this.status.textContent = message; }, () => { if (this.client === client) { this.client = null; this.pending = false; this.updateControls(); } });
      this.client = client; this.media.replaceChildren(client.audio); this.settings.open = false;
      this.status.textContent = "Connecting. The first call downloads and loads models…";
      await client.start(this.selected, mode, settings, this.prompt.value);
    } catch (error) {
      if (generation === this.generation) { await this.stop(); this.showError(error); }
    } finally { if (generation === this.generation) { this.pending = false; this.updateControls(); } }
  }
  private showError(error: unknown): void { this.status.textContent = error instanceof Error ? error.message : String(error); }
  private receive(event: Notice): void {
    if (event.type === "state" || event.type === "error") { this.status.textContent = event.text; return; }
    if (event.type === "interrupted") { const message = this.messages.get(`${event.turn}:assistant`); if (message !== undefined) message.dataset["interrupted"] = "true"; this.status.textContent = "Interrupted · listening"; return; }
    if (event.type === "metrics") {
      this.timing.textContent = this.metrics.update(event);
      return;
    }
    const key = `${event.turn}:${event.speaker}`;
    let message = this.messages.get(key);
    if (message === undefined) {
      const entry = element("article", `conversation-message ${event.speaker}`);
      message = element("p"); entry.append(element("strong", "", event.speaker === "user" ? "You" : "Assistant"), message);
      this.messages.set(key, message); this.feed.append(entry);
    }
    message.textContent = event.final ? event.text : `${message.textContent ?? ""}${event.text}`;
    message.scrollIntoView?.({ block: "nearest" });
  }
  async stop(): Promise<void> {
    this.generation++;
    this.pending = false;
    const client = this.client;
    if (client !== null) { this.status.textContent = "Ending call…"; try { await client.stop(); this.status.textContent = "Call ended"; } catch (error) { this.showError(error); } }
    this.client = null; this.muted = false; this.mute.textContent = "Mute";
    this.updateControls();
  }
  async dispose(): Promise<void> { this.lifetime.abort(); await this.stop(); this.panel?.dispose(); }
}
