import type { ControlPanel } from "../../components/panel.js";
import type { AddedSynthesisModel } from "../../domain/experiment.js";
import { SpeechPanel } from "../synthesis/form.js";
import { voices } from "./voices.js";
import type { Request } from "./request.js";

export function createPanel(model: Extract<AddedSynthesisModel, { id: "phonon" }>): ControlPanel {
  return new SpeechPanel<Request>(model, form => {
    const text = form.text("Text to speak", "Hello. This is Voice Lab.", 2000);
    const voice = form.choice("Voice", voices, "default");
    const advanced = form.section("Sampling");
    const temperature = advanced.decimal("Temperature", 0.3, 0, 2);
    const streaming = advanced.checkbox("Stream while generating", true);
    const seed = form.integer("Seed", 0, 0, 4294967295);
    const max_frames = form.integer("Frame limit", 750, 1, 750);
    return () => ({ text: text(), voice: voice(), temperature: temperature(), streaming: streaming(), seed: seed(), max_frames: max_frames() });
  });
}
