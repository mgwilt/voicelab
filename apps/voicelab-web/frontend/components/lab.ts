import type { Model } from "../domain/experiment.js";
import { SessionRunner, type RunState } from "../domain/runner.js";
import { element } from "./dom.js";
import { Measurements } from "./measurements.js";
import { ModelPicker } from "./model-picker.js";
import type { ControlPanel } from "./panel.js";
import { Transcript } from "./transcript.js";

/** Shared shell: model details enter through the injected panel factory. */
export class Lab {
  readonly element = element("div", "lab");
  private readonly panelHost = element("div", "panel-host");
  private readonly title = element("h2", "experiment-title", "Choose a model");
  private readonly start = element("button", "button-primary", "Start test");
  private readonly stop = element("button", "button-secondary", "Stop");
  private readonly status = element("p", "session-status", "Ready");
  private readonly measurements = new Measurements();
  private readonly transcript = new Transcript();
  private readonly listeners = new AbortController();
  private readonly picker: ModelPicker;
  private readonly runner: SessionRunner;
  private panel: ControlPanel | null = null;
  private model: Model | null = null;
  private selection = 0;
  private switching = false;
  private disposed = false;

  constructor(models: readonly Model[], private readonly createPanel: (model: Model) => ControlPanel) {
    this.status.setAttribute("role", "status");
    this.status.setAttribute("aria-live", "polite");
    this.start.type = "button"; this.stop.type = "button";
    this.start.disabled = true; this.stop.disabled = true;
    this.picker = new ModelPicker(models, model => { void this.select(model); });
    this.runner = new SessionRunner(state => this.renderState(state), event => {
      if (event.type === "measurement") this.measurements.update(event.value);
      else if (event.type === "transcript") this.transcript.update(event.speaker, event.text);
    });
    const choose = element("section", "model-section");
    choose.append(element("h2", "section-title", "Choose a model"), this.picker.element);
    const experiment = element("section", "experiment");
    const actions = element("div", "actions");
    actions.append(this.start, this.stop);
    experiment.append(this.title, this.panelHost, actions, this.status);
    const workspace = element("div", "workspace-grid");
    workspace.append(experiment, this.measurements.element);
    this.element.append(choose, workspace, this.transcript.element);
    this.start.addEventListener("click", () => {
      const panel = this.panel;
      if (panel === null || this.switching || this.runner.busy) return;
      this.measurements.reset(); this.transcript.reset();
      this.runner.start(emit => panel.createSession(emit));
    }, { signal: this.listeners.signal });
    this.stop.addEventListener("click", () => this.runner.stop(), { signal: this.listeners.signal });
    const first = models.find(model => model.available);
    if (first !== undefined) void this.select(first);
    else this.status.textContent = "No model backends are available on this server.";
  }

  async select(model: Model): Promise<void> {
    if (this.disposed || !model.available) return;
    const selection = ++this.selection;
    this.switching = true;
    this.start.disabled = true;
    await this.runner.cancel();
    if (this.disposed || selection !== this.selection) return;
    this.panel?.dispose();
    this.panel = this.createPanel(model);
    this.model = model;
    this.panelHost.replaceChildren(this.panel.element);
    this.title.textContent = model.label;
    this.picker.select(model.id);
    this.start.textContent = model.kind === "synthesis" ? "Generate speech" : "Start test";
    this.measurements.reset(); this.transcript.reset();
    this.transcript.show(model.kind !== "synthesis");
    this.switching = false;
    this.renderState({ phase: "idle", message: "Ready. The first test loads the model.", error: false });
  }

  private renderState(state: RunState): void {
    if (this.disposed) return;
    const active = state.phase !== "idle";
    this.start.disabled = active || this.switching || this.model === null;
    this.stop.disabled = !active;
    this.stop.textContent = state.phase === "stopping" ? "Cancel now" : "Stop";
    this.panel?.setDisabled(active);
    this.status.textContent = state.message;
    this.status.dataset["state"] = state.error ? "error" : state.phase;
    this.element.setAttribute("aria-busy", String(active));
  }

  async dispose(): Promise<void> {
    if (this.disposed) return;
    this.disposed = true;
    this.selection++;
    this.listeners.abort();
    this.picker.dispose();
    await this.runner.dispose();
    this.panel?.dispose();
    this.panel = null;
  }
}
