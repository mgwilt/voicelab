import type { BreezeModel } from "../../domain/session.js";
import { SpeechPanel } from "../synthesis/form.js";
import type { SynthesisRequest } from "./request.js";

export class BreezePanel extends SpeechPanel<SynthesisRequest> {
  constructor(model: BreezeModel) {
    super(model, form => {
      const text = form.text("Text to speak", "Hello. This is a speech test in Voice Lab.", 4000);
      const mode = form.choice("Voice mode", ["design", "clone", "direction"], "design");
      const instruction = form.text("Voice instruction (design or direction)", "A clear, natural conversational voice.", 2000);
      const audio = form.referenceAudio("Reference voice (clone or direction)");
      const transcript = form.optionalText("Exact reference transcript", 4000);
      const advanced = form.section("Generation controls");
      const seed = advanced.integer("Seed", 0, 0, 4294967295);
      const frames = advanced.integer("Frame limit", 750, 1, 750);
      const guidance = advanced.decimal("Guidance", 4, 0.1, 20);
      const temperature = advanced.decimal("Temperature", 0.9, 0.01, 2);
      const topK = advanced.integer("Top K", 50, 0, 2048);
      const topP = advanced.decimal("Top P", 1, 0.01, 1);
      form.note("Cloning preserves the reference voice; direction also applies your instruction. A frame-limit cutoff is incomplete.");
      return async () => {
        const selected = mode();
        const refAudio = await audio();
        const refText = transcript();
        if (selected !== "design" && (refAudio === null || refText === null)) throw new Error("Cloning and direction require reference audio and its exact transcript.");
        if (selected === "design" && (refAudio !== null || refText !== null)) throw new Error("Select clone or direction to use a reference.");
        return {
          text: text(), instruction: selected === "clone" ? null : instruction(),
          ref_audio: refAudio, ref_text: refText,
          seed: seed(), max_frames: frames(), cfg_scale: guidance(),
          temperature: temperature(), top_k: topK(), top_p: topP(),
        };
      };
    });
  }
}
