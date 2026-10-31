import type { Model } from "../domain/experiment.js";
import { SessionRunner, type RunState } from "../domain/runner.js";
import { element } from "./dom.js";
import { Measurements } from "./measurements.js";
import { ModelPicker } from "./model-picker.js";
import type { ControlPanel } from "./panel.js";
import { Transcript } from "./transcript.js";
import { Workbench } from "./workbench.js";

/** Shared shell: model details enter through the injected panel factory. */
export class Lab {
  readonly element = element("div", "lab");
  private readonly panelHost = element("div", "panel-host");
  private readonly title = element("h2", "experiment-title", "Choose a model");
  private readonly description = element("p", "model-description");
  private readonly backend = element("span", "runtime-label");
  private readonly phase = element("span", "session-phase", "Ready");
  private readonly start = element("button", "button-primary", "Start test");
  private readonly stop = element("button", "button-secondary", "Stop");
  private readonly status = element("p", "session-status", "Ready");
  private readonly measurements = new Measurements();
  private readonly transcript = new Transcript();
  private readonly listeners = new AbortController();
  private readonly picker: ModelPicker;
  private readonly runner: SessionRunner;
  private readonly workbench: Workbench;
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
    const heading = element("header", "experiment-header");
    heading.append(this.title, this.backend, this.description);
    const controls = element("section", "workbench-content controls-view");
    controls.append(this.panelHost);
    const output = element("section", "workbench-content output-view");
    output.append(this.measurements.element, this.transcript.element);
    this.workbench = new Workbench(controls, output);
    const runbar = element("section", "runbar");
    runbar.setAttribute("aria-label", "Session controls");
    const readout = element("div", "runbar-readout");
    const outputLink = element("button", "output-link");
    outputLink.type = "button";
    outputLink.title = "View measurements and transcript";
    outputLink.append(element("span", "visually-hidden", "View output: "), this.measurements.summary);
    outputLink.addEventListener("click", () => {
      this.workbench.show("output");
      output.focus();
    }, { signal: this.listeners.signal });
    readout.append(this.phase, outputLink);
    const actions = element("div", "actions run-actions");
    actions.append(this.start, this.stop);
    runbar.append(readout, actions, this.status);
    this.element.append(this.picker.element, heading, this.workbench.element, runbar);
    this.start.addEventListener("click", () => {
      const panel = this.panel;
      if (panel === null || this.switching || this.runner.busy) return;
      this.workbench.show("controls");
      this.measurements.reset(); this.transcript.reset();
      this.runner.start(emit => panel.createSession(emit));
    }, { signal: this.listeners.signal });
    this.stop.addEventListener("click", () => this.runner.stop(), { signal: this.listeners.signal });
    const first = models.find(model => model.available);
    if (first !== undefined) void this.select(first);
    else {
      this.phase.textContent = "Unavailable";
      this.status.textContent = "No model backends are available on this server.";
      this.picker.element.open = true;
    }
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
    this.description.textContent = model.description;
    this.backend.textContent = model.backend;
    this.workbench.show("controls");
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
    this.phase.textContent = state.error ? "Error" : {
      idle: "Ready", preparing: "Preparing", running: "Running", draining: "Playing", stopping: "Stopping",
    }[state.phase];
    this.phase.dataset["state"] = state.error ? "error" : state.phase;
    this.status.textContent = state.message;
    this.status.dataset["state"] = state.error ? "error" : state.phase;
    this.panelHost.setAttribute("aria-busy", String(active));
  }

  async dispose(): Promise<void> {
    if (this.disposed) return;
    this.disposed = true;
    this.selection++;
    this.listeners.abort();
    this.picker.dispose();
    this.workbench.dispose();
    await this.runner.dispose();
    this.panel?.dispose();
    this.panel = null;
  }
}
