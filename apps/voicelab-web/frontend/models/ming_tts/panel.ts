import type { ControlPanel } from "../../components/panel.js";
import type { AddedSynthesisModel } from "../../domain/session.js";
import { SpeechPanel } from "../synthesis/form.js";
import type { Request } from "./request.js";
import { voices } from "./voices.js";

function optional<Value extends string>(value: Value | ""): Value | null {
  return value === "" ? null : value;
}
export function createPanel(model: Extract<AddedSynthesisModel, { id: "ming_tts" }>): ControlPanel {
  return new SpeechPanel<Request>(model, form => {
    const text = form.text("Text to speak", "Hello. This is Voice Lab.", 500);
    const language = form.choice("Language", ["zh", "en", "ja", "ko", "de", "fr", "es"], "en");
    const voice = form.choice("Built-in voice (blank uses other conditioning)", voices, "");
    const reference = form.section("Clone a reference voice");
    const ref_audio = reference.referenceAudio("Reference speech");
    const ref_text = reference.optionalText("Exact reference transcript", 500);
    const trim_reference_silence = reference.checkbox("Trim trailing silence from acoustic prompt", true);
    const design = form.section("Voice design and delivery");
    design.note("Use a freeform instruction alone, or structured controls. A built-in voice, voice design, and reference voice are separate choices.");
    const instruction = design.optionalText("Freeform delivery instruction", 500);
    const style = design.optionalText("Voice design description", 500);
    const emotion = design.choice("Emotion", ["", "高兴", "悲伤", "愤怒", "恐惧", "厌恶", "惊讶", "中性"], "");
    const dialect = design.choice("Dialect", ["", "cantonese", "chuanyu dialect"], "");
    const speed = design.choice("Speed", ["", "慢速", "中速", "快速"], "");
    const pitch = design.choice("Pitch", ["", "低", "中", "高"], "");
    const volume = design.choice("Volume", ["", "低", "中", "高"], "");
    const advanced = form.section("Sampling");
    const seed = advanced.integer("Seed", 0, 0, 4294967295);
    const max_frames = advanced.integer("Frame limit per segment", 120, 1, 120);
    const cfg_scale = advanced.decimal("Guidance", 2, 0.01, 10);
    const flow_steps = advanced.integer("Flow steps", 10, 1, 100);
    const temperature = advanced.decimal("Temperature", 0, 0, 2);
    const sigma = advanced.decimal("Noise sigma", 0.25, 0, 1);
    return async () => ({ text: text(), language: language(), voice: optional(voice()),
      ref_audio: await ref_audio(), ref_text: ref_text(), trim_reference_silence: trim_reference_silence(), instruction: instruction(), style: style(),
      emotion: optional(emotion()), dialect: optional(dialect()), speed: optional(speed()),
      pitch: optional(pitch()), volume: optional(volume()), seed: seed(), max_frames: max_frames(),
      cfg_scale: cfg_scale(), flow_steps: flow_steps(), temperature: temperature(), sigma: sigma() });
  });
}
