import type { ControlPanel } from "../components/panel.js";
import type { Model } from "../domain/experiment.js";
import { unreachable } from "../domain/experiment.js";
import { BreezePanel } from "./breeze/panel.js";
import { DuplexPanel } from "./nemotron/panel.js";
import { SynthesisPanel } from "./sesame/panel.js";

export function createPanel(model: Model): ControlPanel {
  switch (model.id) {
    case "nemotron": return new DuplexPanel(model);
    case "sesame": return new SynthesisPanel(model);
    case "breeze": return new BreezePanel(model);
    default: return unreachable(model);
  }
}
