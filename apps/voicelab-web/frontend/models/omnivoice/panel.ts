import type { ControlPanel } from "../../components/panel.js";
import type { AddedSynthesisModel } from "../../domain/session.js";
import { SpeechPanel } from "../synthesis/form.js";
import type { Request } from "./request.js";

export function createPanel(model: Extract<AddedSynthesisModel, { id: "omnivoice" }>): ControlPanel {
  return new SpeechPanel<Request>(model, form => {
    const text = form.text("Text to speak", "Hello. This is Voice Lab.", 4000);
    const reference = form.section("Reference voice (optional)");
    const refAudio = reference.referenceAudio("Reference WAV");
    const refText = reference.optionalText("Reference transcript", 4000);
    const language = form.optionalText("Language (blank: language agnostic)", 80);
    const instruction = form.optionalText("Instruction", 1000);
    const steps = form.integer("Steps", 32, 1, 100);
    const speed = form.decimal("Speed", 1.0, 0.5, 2.0);
    const seed = form.integer("Seed", 0, 0, 4294967295);
    const advanced = form.section("Advanced generation");
    advanced.note("Text normalization needs optional upstream dependencies. English and Chinese require WeTextProcessing, which is not installed by default.");
    const duration = advanced.optionalDecimal("Duration", 0.001, 120);
    const normalize = advanced.checkbox("Normalize", false);
    const denoise = advanced.checkbox("Denoise", true);
    const preprocess_prompt = advanced.checkbox("Preprocess prompt", true);
    const postprocess_output = advanced.checkbox("Postprocess output", true);
    const guidance_scale = advanced.decimal("Guidance scale", 2.0, 0, 10);
    const t_shift = advanced.decimal("T shift", 0.1, 0.001, 5);
    const layer_penalty_factor = advanced.decimal("Layer penalty factor", 5.0, 0, 20);
    const position_temperature = advanced.decimal("Position temperature", 5.0, 0, 20);
    const class_temperature = advanced.decimal("Class temperature", 0.0, 0, 5);
    const audio_chunk_duration = advanced.decimal("Audio chunk duration", 15.0, 0, 120);
    const audio_chunk_threshold = advanced.decimal("Audio chunk threshold", 30.0, 0, 120);
    const pad_duration = advanced.decimal("Pad duration", 0.1, 0, 2);
    const fade_duration = advanced.decimal("Fade duration", 0.1, 0, 2);
    return async () => {
      const audio = await refAudio();
      const transcript = refText();
      if (audio === null && transcript !== null) throw new Error("A reference transcript requires reference audio.");
      return {
        text: text(),
        language: language(),
        instruction: instruction(),
        steps: steps(),
        speed: speed(),
        ref_audio: audio,
        ref_text: transcript,
        duration: duration(),
        normalize: normalize(),
        denoise: denoise(),
        preprocess_prompt: preprocess_prompt(),
        postprocess_output: postprocess_output(),
        guidance_scale: guidance_scale(),
        t_shift: t_shift(),
        layer_penalty_factor: layer_penalty_factor(),
        position_temperature: position_temperature(),
        class_temperature: class_temperature(),
        audio_chunk_duration: audio_chunk_duration(),
        audio_chunk_threshold: audio_chunk_threshold(),
        pad_duration: pad_duration(),
        fade_duration: fade_duration(),
        seed: seed(),
      };
    };
  });
}
