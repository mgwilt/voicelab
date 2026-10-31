import type { ControlPanel } from "../../components/panel.js";
import type { AddedSynthesisModel } from "../../domain/experiment.js";
import { SpeechPanel } from "../synthesis/form.js";
import { voices } from "./voices.js";
import type { Request } from "./request.js";

export function createPanel(model: Extract<AddedSynthesisModel, { id: "kokoro" }>): ControlPanel {
  return new SpeechPanel<Request>(model, form => {
    const text = form.text("Text to speak", "Hello. This is Voice Lab.", 2000);
    const voice = form.choice("Voice (language follows its prefix)", voices, "af_heart");
    const blend = form.optionalText("Blend additional voice IDs (same language, comma-separated)", 400);
    const speed = form.decimal("Speed", 1, 0.5, 2);
    return () => ({ text: text(), voice: blend() === null ? voice() : `${voice()},${blend()}`, speed: speed() });
  });
}
