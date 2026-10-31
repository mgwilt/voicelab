import type { ControlPanel } from "../../components/panel.js";
import type { AddedSynthesisModel } from "../../domain/session.js";
import { SpeechPanel } from "../synthesis/form.js";
import type { Request } from "./request.js";

export function createPanel(model: Extract<AddedSynthesisModel, { id: "maya" }>): ControlPanel {
  return new SpeechPanel<Request>(model, form => {
    const text = form.text("Text to speak", "Hello. This is Voice Lab.", 4000);
    const reference = form.section("Reference voice (optional)");
    const refAudio = reference.referenceAudio("Reference WAV");
    const refText = reference.optionalText("Reference transcript", 4000);
    const language = form.choice("Language", ["as", "bn", "en", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"], "en");
    const voice = form.choice("Named voice (ignored with reference)", ["Ahana", "Vaidehi", "Malini", "Charvi", "Jahnavi", "Anirban", "Nakul", "Vasant", "Siddhant", "Satvik", "Abhay", "Nivedita", "Ishwari", "Ritvik", "Vrinda", "Sreekar", "Kayal", "Shaunak", "Akshat", "Tanushree"], "Siddhant");
    const max_frames = form.integer("Max frames", 750, 1, 1500);
    const seed = form.integer("Seed", 0, 0, 4294967295);
    const advanced = form.section("Advanced generation");
    const ref_language = advanced.choice("Reference language", ["default", "as", "bn", "en", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"], "default");
    const conditioning = advanced.choice("Conditioning", ["icl", "spk", "icl_spk"], "icl_spk");
    const temperature = advanced.decimal("Temperature", 0.9, 0, 2);
    const top_k = advanced.integer("Top k", 64, 0, 2048);
    const top_p = advanced.decimal("Top p", 1.0, 0.001, 1);
    const depth_temperature = advanced.decimal("Depth temperature", 0.7, 0, 2);
    const depth_top_k = advanced.integer("Depth top k", 64, 0, 2048);
    const presence_penalty = advanced.decimal("Presence penalty", 0.8, 0, 5);
    const presence_window = advanced.integer("Presence window", 12, 0, 512);
    const min_frames = advanced.integer("Min frames", 2, 0, 1500);
    const streaming_interval = advanced.decimal("Streaming interval", 0.32, 0.08, 5);
    const normalize = advanced.checkbox("Normalize", true);
    return async () => {
      const audio = await refAudio();
      const transcript = refText();
      if (audio === null && transcript !== null) throw new Error("A reference transcript requires reference audio.");
      const referenceLanguage = ref_language();
      return {
        text: text(),
        language: language(),
        voice: audio === null ? voice() : null,
        ref_audio: audio,
        ref_text: transcript,
        ref_language: referenceLanguage === "default" ? null : referenceLanguage,
        conditioning: conditioning(),
        temperature: temperature(),
        top_k: top_k(),
        top_p: top_p(),
        depth_temperature: depth_temperature(),
        depth_top_k: depth_top_k(),
        presence_penalty: presence_penalty(),
        presence_window: presence_window(),
        min_frames: min_frames(),
        streaming_interval: streaming_interval(),
        max_frames: max_frames(),
        seed: seed(),
        normalize: normalize(),
      };
    };
  });
}
