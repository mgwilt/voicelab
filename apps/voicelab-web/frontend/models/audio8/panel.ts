import type { ControlPanel } from "../../components/panel.js";
import type { AddedSynthesisModel } from "../../domain/experiment.js";
import { SpeechPanel } from "../synthesis/form.js";
import type { Request } from "./request.js";

export function createPanel(model: Extract<AddedSynthesisModel, { id: "audio8" }>): ControlPanel {
  return new SpeechPanel<Request>(model, form => {
    const text = form.text("Text to speak", "Hello. This is Voice Lab.", 2000);
    const reference = form.section("Clone a reference voice");
    const ref_audio = reference.referenceAudio("Reference speech");
    const ref_text = reference.optionalText("Exact reference transcript", 2000);
    const sampling = form.section("Sampling");
    const top_p = sampling.decimal("Top p", 0.9, 0.01, 1);
    const top_k = sampling.integer("Top k (0 disables filtering)", 50, 0, 4096);
    const do_sample = sampling.checkbox("Sample tokens", true);
    const seed = form.integer("Seed", 0, 0, 4294967295);
    const temperature = form.decimal("Temperature", 0.7, 0.01, 2);
    const max_frames = form.integer("Frame limit", 512, 1, 1024);
    return async () => ({ ref_audio: await ref_audio(), ref_text: ref_text(), top_p: top_p(), top_k: top_k(), do_sample: do_sample(), text: text(), seed: seed(), temperature: temperature(), max_frames: max_frames() });
  });
}
