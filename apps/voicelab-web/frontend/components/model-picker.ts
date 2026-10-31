import type { Model, ModelId } from "../domain/experiment.js";
import { element } from "./dom.js";

export class ModelPicker {
  readonly element = element("div", "model-grid");
  private readonly listeners = new AbortController();
  private readonly choices = new Map<ModelId, HTMLButtonElement>();

  constructor(models: readonly Model[], select: (model: Model) => void) {
    this.element.setAttribute("aria-label", "Choose a voice model");
    for (const model of models) {
      const button = element("button", "model-card");
      button.type = "button";
      button.setAttribute("aria-pressed", "false");
      button.disabled = !model.available;
      const label = element("span", "model-card-title", model.label);
      const kind = element("span", "tag", model.kind === "duplex" ? "Live conversation" : "Text to speech");
      button.append(label, kind, element("span", "model-description", model.description),
        element("span", "model-backend", model.available ? model.backend : model.reason ?? "Unavailable"));
      button.addEventListener("click", () => select(model), { signal: this.listeners.signal });
      this.choices.set(model.id, button);
      this.element.append(button);
    }
  }
  select(id: ModelId): void {
    for (const [key, button] of this.choices) button.setAttribute("aria-pressed", String(key === id));
  }
  dispose(): void { this.listeners.abort(); }
}
