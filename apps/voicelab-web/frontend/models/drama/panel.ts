import type { ControlPanel } from "../../components/panel.js";
import type { AddedSynthesisModel } from "../../domain/experiment.js";
import { SpeechPanel } from "../synthesis/form.js";
import type { Request, Reference } from "./request.js";

export function createPanel(model: Extract<AddedSynthesisModel, { id: "drama" }>): ControlPanel {
  return new SpeechPanel<Request>(model, form => {
    const text = form.text("Text to speak", "Hello. This is Voice Lab.", 4000);
    const refs = form.section("Reference voices (optional)");
    const references = Array.from({ length: 4 }, (_, index) => ({
      audio: refs.referenceAudio(`Reference ${index + 1} audio`),
      text: refs.optionalText(`Reference ${index + 1} transcript`, 4000),
    }));
    const seed = form.integer("Seed", 0, 0, 4294967295);
    const max_new_tokens = form.integer("Max new tokens", 1024, 1, 2048);
    const history = form.section("Earlier takes for continuation editing");
    history.note("Supply the previous audio and exact transcript. The model generates a new take with this history; it does not splice a recording.");
    const takes = Array.from({ length: 2 }, (_, index) => ({
      audio: history.referenceAudio(`Earlier take ${index + 1} audio`),
      text: history.optionalText(`Earlier take ${index + 1} transcript`, 4000),
    }));
    const advanced = form.section("Advanced generation");
    const top_k = advanced.optionalInteger("Top k", 1, 1024);
    const streaming_interval = advanced.decimal("Streaming interval", 0.32, 0.05, 5);
    const first_chunk_frames = advanced.optionalInteger("First chunk frames", 1, 256);
    const temperature = advanced.decimal("Temperature", 1.0, 0.001, 2);
    const top_p = advanced.decimal("Top p", 0.9, 0.001, 1);
    return async () => {
      const paired = await Promise.all(references.map(async (reference): Promise<Reference | null> => {
        const audio = await reference.audio();
        const transcript = reference.text();
        if (audio === null && transcript === null) return null;
        if (audio === null || transcript === null) throw new Error("Each reference needs audio and its transcript.");
        return { audio, text: transcript };
      }));
      const previous = await Promise.all(takes.map(async (take): Promise<Reference | null> => {
        const audio = await take.audio();
        const transcript = take.text();
        if (audio === null && transcript === null) return null;
        if (audio === null || transcript === null) throw new Error("Each earlier take needs audio and its transcript.");
        return { audio, text: transcript };
      }));
      const voiceReferences = paired.filter((item): item is Reference => item !== null);
      const previousTakes = previous.filter((item): item is Reference => item !== null);
      if (voiceReferences.length + previousTakes.length > 4) throw new Error("Use at most four recordings across references and earlier takes.");
      return {
        text: text(),
        references: voiceReferences,
        previous_takes: previousTakes,
        top_k: top_k(),
        streaming_interval: streaming_interval(),
        first_chunk_frames: first_chunk_frames(),
        seed: seed(),
        max_new_tokens: max_new_tokens(),
        temperature: temperature(),
        top_p: top_p(),
      };
    };
  });
}
