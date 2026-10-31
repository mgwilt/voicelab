import type { ControlPanel } from "../../components/panel.js";
import type { AddedSynthesisModel } from "../../domain/experiment.js";
import { SpeechPanel } from "../synthesis/form.js";
import type { Request } from "./request.js";

export function createPanel(model: Extract<AddedSynthesisModel, { id: "kitten" }>): ControlPanel {
  return new SpeechPanel<Request>(model, form => {
    const text = form.text("Text to speak", "Hello. This is Voice Lab.", 4000);
    const reference = form.section("Reference voice (optional)");
    const refAudio = reference.referenceAudio("Reference WAV");
    const refText = reference.optionalText("Reference transcript", 4000);
    const voice = form.text("Named voice (ignored with reference)", "Bella", 100);
    const preset = form.choice("Preset", ["stable", "expressive"], "stable");
    const max_tokens = form.integer("Max tokens", 1000, 1, 2000);
    const seed = form.integer("Seed", 0, 0, 4294967295);
    const advanced = form.section("Advanced generation");
    const use_reference_prompt = advanced.choice("Use reference prompt", ["default", "enabled", "disabled"], "default");
    const temperature = advanced.optionalDecimal("Temperature", 0.001, 2);
    const top_k = advanced.optionalInteger("Top k", 0, 1024);
    const top_p = advanced.optionalDecimal("Top p", 0.001, 1);
    const min_p = advanced.optionalDecimal("Min p", 0, 1);
    const repetition_penalty = advanced.decimal("Repetition penalty", 1.1, 1, 3);
    const repetition_window = advanced.optionalInteger("Repetition window", 1, 2000);
    const token_run_penalty = advanced.decimal("Token run penalty", 1.3, 1, 5);
    const token_run_grace = advanced.integer("Token run grace", 10, 1, 100);
    const chunk_chars = advanced.optionalInteger("Chunk chars", 1, 4000);
    const chunk_min_chars = advanced.optionalInteger("Chunk min chars", 0, 4000);
    const use_emotion = advanced.choice("Use emotion", ["default", "enabled", "disabled"], "default");
    const normalize = advanced.checkbox("Normalize", false);
    return async () => {
      const audio = await refAudio();
      const transcript = refText();
      if (audio === null && transcript !== null) throw new Error("A reference transcript requires reference audio.");
      return {
        text: text(),
        voice: audio === null ? voice() : null,
        preset: preset(),
        max_tokens: max_tokens(),
        ref_audio: audio,
        ref_text: transcript,
        use_reference_prompt: use_reference_prompt() === "default" ? null : use_reference_prompt() === "enabled",
        temperature: temperature(),
        top_k: top_k(),
        top_p: top_p(),
        min_p: min_p(),
        repetition_penalty: repetition_penalty(),
        repetition_window: repetition_window(),
        token_run_penalty: token_run_penalty(),
        token_run_grace: token_run_grace(),
        chunk_chars: chunk_chars(),
        chunk_min_chars: chunk_min_chars(),
        use_emotion: use_emotion() === "default" ? null : use_emotion() === "enabled",
        seed: seed(),
        normalize: normalize(),
      };
    };
  });
}
