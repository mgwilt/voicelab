import type { Model, ModelId } from "../domain/session.js";
import { element, labeled } from "./dom.js";

const workflowLabels: Record<Model["kind"], string> = {
  duplex: "Live conversation", synthesis: "Text to speech", conversation: "Audio and text",
};

/** A compact explorer; unavailable models remain discoverable with their reason. */
export class ModelPicker {
  readonly element = element("details", "model-explorer");
  private readonly summary = element("summary", "explorer-toggle");
  private readonly selected = element("span", "explorer-selection", "Choose a model");
  private readonly search = element("input", "model-search");
  private readonly count = element("p", "explorer-count");
  private readonly empty = element("p", "field-hint", "No matching models. Try a different name or workflow.");
  private readonly listeners = new AbortController();
  private readonly choices = new Map<ModelId, { model: Model; button: HTMLButtonElement; row: HTMLLIElement }>();

  constructor(models: readonly Model[], select: (model: Model) => void) {
    this.summary.append(element("span", "explorer-label", "Models"), this.selected,
      element("span", "explorer-chevron", "⌄"));
    this.summary.lastElementChild?.setAttribute("aria-hidden", "true");
    this.element.append(this.summary);
    const content = element("div", "explorer-content");
    this.search.type = "search";
    this.search.placeholder = "Name, workflow, or runtime";
    this.search.autocomplete = "off";
    this.search.spellcheck = false;
    this.count.setAttribute("role", "status");
    const list = element("ul", "model-list");
    list.setAttribute("aria-label", "Choose a voice model");
    for (const model of models) {
      const row = element("li", "model-row");
      const button = element("button", "model-choice");
      button.type = "button";
      button.setAttribute("aria-pressed", "false");
      button.disabled = !model.available;
      button.append(element("span", "model-choice-title", model.label),
        element("span", "tag", workflowLabels[model.kind]),
        element("span", "model-backend", model.available ? model.backend : "Unavailable"));
      button.addEventListener("click", () => select(model), { signal: this.listeners.signal });
      row.append(button);
      if (!model.available) row.append(element("p", "unavailable-reason", model.reason ?? "No runtime is available on this server."));
      this.choices.set(model.id, { model, button, row });
      list.append(row);
    }
    content.append(labeled("Find a model", this.search), this.count, list, this.empty);
    this.element.append(content);
    this.search.addEventListener("input", () => this.filter(), { signal: this.listeners.signal });
    this.element.addEventListener("keydown", event => {
      if (event.key === "Escape" && this.element.open) {
        event.preventDefault();
        this.element.open = false;
        this.summary.focus();
      }
    }, { signal: this.listeners.signal });
    this.filter();
  }

  private filter(): void {
    const query = this.search.value.trim().toLocaleLowerCase();
    let count = 0;
    for (const { model, row } of this.choices.values()) {
      row.hidden = !`${model.label} ${workflowLabels[model.kind]} ${model.backend}`.toLocaleLowerCase().includes(query);
      if (!row.hidden) count++;
    }
    this.count.textContent = `${count} ${count === 1 ? "model" : "models"}`;
    this.empty.hidden = count > 0;
  }

  select(id: ModelId): void {
    const focusInside = this.element.contains(document.activeElement);
    for (const [key, { model, button }] of this.choices) {
      button.setAttribute("aria-pressed", String(key === id));
      if (key === id) this.selected.textContent = model.label;
    }
    this.element.open = false;
    this.search.value = "";
    this.filter();
    if (focusInside) this.summary.focus();
  }

  dispose(): void { this.listeners.abort(); }
}
