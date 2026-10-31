import { element, labeled } from "../components/dom.js";
import type { ControlPanel } from "../components/panel.js";
import { ModelPicker } from "../components/model-picker.js";
import { Workbench } from "../components/workbench.js";
import type { Model } from "../domain/session.js";
import { ConversationClient, type ConversationConnection, type ConnectionFactory } from "./client.js";
import { MicrophoneStatus } from "./microphone-status.js";
import { SpeakerControl } from "./speaker-control.js";
import { ConversationMetrics } from "./metrics.js";
import type { Notice } from "./events.js";

export class ConversationView {
  readonly element = element("section", "lab conversation");
  private readonly picker: ModelPicker;
  private readonly title = element("h2", "session-title");
  private readonly description = element("p", "model-description");
  private readonly backend = element("span", "runtime-label");
  private readonly workbench: Workbench;
  private reusableClient: ConversationConnection | null = null;
  private readonly mode = element("select");
  private readonly settings = element("section", "workbench-content controls-view");
  private readonly panelHost = element("div");
  private readonly prompt = element("textarea");
  private readonly feed = element("div", "conversation-feed");
  private readonly status = element("p", "session-status", "Choose a voice, then start a conversation.");
  private readonly metrics = new ConversationMetrics();
  private readonly microphone = new MicrophoneStatus();
  private readonly timing = element("p", "conversation-timing");
  private readonly start = element("button", "button-primary", "Start conversation");
  private readonly end = element("button", "button-secondary", "End call");
  private readonly mute = element("button", "button-secondary", "Mute microphone");
  private readonly speaker = new SpeakerControl();
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

  constructor(models: readonly Model[], private readonly createPanel: (model: Model) => ControlPanel, private readonly connection: ConnectionFactory = (event, status, ended, processing) => new ConversationClient(event, status, ended, undefined, processing)) {
    this.picker = new ModelPicker(models.map(model => ({ ...model, available: model.available && (model.conversationModes?.length ?? 0) > 0 })), model => { void this.select(model); });
    this.prompt.value = "Answer briefly and naturally. Use plain spoken language.";
    this.prompt.maxLength = 2000; this.prompt.rows = 3;
    this.settings.append(element("h3", "section-title", "Conversation controls"), labeled("Conversation mode", this.mode), labeled("Assistant instructions (cascade)", this.prompt), this.panelHost);
    this.feed.setAttribute("role", "log"); this.feed.setAttribute("aria-live", "polite");
    this.status.setAttribute("role", "status");
    const actions = element("div", "conversation-actions"); actions.append(this.start, this.mute, this.interrupt, this.end);
    const footer = element("div", "runbar conversation-footer"); footer.append(this.status, actions);
    const heading = element("header", "session-header"); heading.append(this.title, this.backend, this.description);
    const output = element("section", "workbench-content output-view");
    output.append(element("h3", "section-title", "Conversation"), this.microphone.element, this.media, this.feed, this.timing, this.metrics.element);
    this.workbench = new Workbench(this.settings, output);
    this.workbench.show("output");
    this.element.append(this.picker.element, heading, this.workbench.element, footer);
    this.end.disabled = this.mute.disabled = this.interrupt.disabled = true;
    const signal = this.lifetime.signal;
    this.mode.addEventListener("change", () => this.updateControls(), { signal });
    this.start.addEventListener("click", () => { void this.begin(); }, { signal });
    this.end.addEventListener("click", () => { void this.stop(); }, { signal });
    this.mute.addEventListener("click", () => { this.muted = !this.muted; this.client?.mute(this.muted); this.updateMicrophoneControl(); }, { signal });
    this.interrupt.addEventListener("click", () => { void this.client?.interrupt().catch(error => this.showError(error)); }, { signal });
    const first = models.find(model => model.available && model.id === "kokoro" && (model.conversationModes?.length ?? 0) > 0) ?? models.find(model => model.available && (model.conversationModes?.length ?? 0) > 0);
    if (first !== undefined) { void this.select(first); }
    else { this.start.disabled = true; this.status.textContent = "No conversation runtimes are available. Open Direct run for other workflows."; }
  }
  private async select(model: Model): Promise<void> {
    this.selected = model;
    this.picker.select(model.id);
    await this.stop();
    if (this.lifetime.signal.aborted || this.selected !== model) return;
    this.panel?.dispose(); this.panel = null; this.panelHost.replaceChildren(); this.mode.replaceChildren();
    this.title.textContent = model.label; this.description.textContent = model.description; this.backend.textContent = model.backend;
    this.panel = this.createPanel(model); this.panel.prepareConversation?.(); this.panelHost.append(this.panel.element);
    const modes = model.conversationModes ?? [];
    for (const mode of modes) { const option = element("option", "", { cascade: "Local voice cascade", turn: "Turn-based speech", duplex: "Live duplex" }[mode]); option.value = mode; this.mode.append(option); }
    this.mode.value = modes[0] ?? "cascade";
    this.feed.replaceChildren(); this.messages.clear(); this.timing.textContent = ""; this.metrics.reset();
    this.updateControls();
  }
  private updateControls(): void {
    const busy = this.client !== null || this.pending;
    if (this.client === null) { this.speaker.bind(null); this.muted = false; this.updateMicrophoneControl(); }
    this.element.dataset["active"] = String(busy);
    this.start.disabled = busy || this.selected === null;
    this.start.hidden = busy;
    this.end.hidden = this.mute.hidden = this.interrupt.hidden = !busy;
    this.media.hidden = !busy;
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
    this.pending = true; this.microphone.update(null); this.updateControls();
    try {
      const settings = await this.panel.conversationConfig();
      if (generation !== this.generation || !this.pending || this.lifetime.signal.aborted) return;
      const mode = this.mode.value;
      if (mode !== "cascade" && mode !== "turn" && mode !== "duplex") throw new Error("Invalid conversation mode");
      this.feed.replaceChildren(); this.messages.clear(); this.timing.textContent = ""; this.metrics.reset();
      const client = this.reusableClient ??= this.connection(event => this.receive(event), message => { this.status.textContent = message; }, () => { this.client = null; this.pending = false; this.updateControls(); }, report => this.microphone.update(report));
      this.client = client; this.speaker.bind(client.audio); this.media.replaceChildren(client.audio, this.speaker.element); this.workbench.show("output");
      this.status.textContent = "Connecting. The first call downloads and loads models…";
      await client.start(this.selected, mode, settings, this.prompt.value);
    } catch (error) {
      if (generation === this.generation) { await this.stop(); this.showError(error); }
    } finally { if (generation === this.generation) { this.pending = false; this.updateControls(); } }
  }
  private showError(error: unknown): void { this.status.textContent = error instanceof Error ? error.message : String(error); }
  private updateMicrophoneControl(): void {
    this.mute.textContent = this.muted ? "Unmute microphone" : "Mute microphone";
    this.mute.setAttribute("aria-pressed", String(this.muted));
  }
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
    message.textContent = event.final || event.textMode === "snapshot" ? event.text : `${message.textContent ?? ""}${event.text}`;
    message.scrollIntoView?.({ block: "nearest" });
  }
  async stop(): Promise<void> {
    this.generation++;
    this.pending = false;
    const client = this.client;
    if (client !== null) { this.status.textContent = "Ending call…"; try { await client.stop(); this.status.textContent = "Call ended"; } catch (error) { this.showError(error); } }
    this.client = null;
    this.updateControls();
  }
  async dispose(): Promise<void> { this.lifetime.abort(); await this.stop(); this.speaker.dispose(); this.panel?.dispose(); this.picker.dispose(); this.workbench.dispose(); }
}
