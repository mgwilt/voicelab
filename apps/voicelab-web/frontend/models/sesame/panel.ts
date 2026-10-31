import type { SesameModel } from "../../domain/session.js";
import { SpeechPanel } from "../synthesis/form.js";
import type { ContextSegment, SynthesisRequest } from "./request.js";

export class SynthesisPanel extends SpeechPanel<SynthesisRequest> {
  constructor(model: SesameModel) {
    super(model, form => {
      const text = form.text("Text to speak", "Hello. This is a speech test in Voice Lab.", 8000);
      const speaker = form.choice("Speaker role", ["0", "1"], "0");
      const audio = form.referenceAudio("Reference speech");
      const transcript = form.optionalText("Exact reference transcript", 8000);
      const context = form.section("Prior conversation (instead of one voice reference)");
      context.note("Supply up to four ordered turns with their original audio, transcript, and speaker role. Combined audio must not exceed 30 seconds.");
      const turns = Array.from({ length: 4 }, (_, index) => {
        const turn = context.section(`Turn ${index + 1}`);
        const turnAudio = turn.referenceAudio("Turn recording");
        const turnText = turn.optionalText("Exact transcript", 8000);
        const turnSpeaker = turn.choice("Speaker role", ["0", "1"], "0");
        return async (): Promise<ContextSegment | null> => {
          const audio = await turnAudio();
          const text = turnText();
          if (audio === null && text === null) return null;
          if (audio === null || text === null) throw new Error("Each context turn needs audio and its transcript.");
          return { audio, text, speaker: turnSpeaker() === "0" ? 0 : 1 };
        };
      });
      const advanced = form.section("Generation controls");
      const seed = advanced.integer("Seed", 42, 0, 4294967295);
      const frames = advanced.integer("Frame limit", 250, 4, 1500);
      const temperature = advanced.decimal("Temperature", 0.9, 0.01, 2);
      const topK = advanced.integer("Top K", 50, 0, 2048);
      form.note("Speaker IDs select conversation roles. Reference audio conditions the voice. Each frame represents 80 ms; a capped output is incomplete.");
      return async () => {
        const refAudio = await audio();
        const refText = transcript();
        if ((refAudio === null) !== (refText === null)) throw new Error("Supply reference audio and its transcript together.");
        const history = (await Promise.all(turns.map(read => read()))).filter(turn => turn !== null);
        if (history.length > 0 && refAudio !== null) throw new Error("Use prior conversation or a single reference voice.");
        return {
          text: text(), speaker: speaker() === "0" ? 0 : 1,
          ref_audio: refAudio, ref_text: refText,
          seed: seed(), max_frames: frames(), temperature: temperature(), top_k: topK(),
          context: history,
        };
      };
    });
  }
}
