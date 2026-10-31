import type { Measurement } from "../../domain/experiment.js";
import { boolean, finite, nullableNumber, parse, record, text } from "../../transport/decode.js";
export type Message =
  | { readonly type: "loading" | "ready" }
  | { readonly type: "assistant_text_delta"; readonly text: string }
  | { readonly type: "done"; readonly value: Measurement; readonly complete: boolean }
  | { readonly type: "error"; readonly message: string };
export function decode(raw: string): Message {
  const data = parse(raw);
  switch (data["type"]) {
    case "loading": return { type: "loading" };
    case "ready":
      if (data["input_sample_rate"] !== 16000 || data["output_sample_rate"] !== 24000) throw new Error("Unexpected Qwen audio rates");
      return { type: "ready" };
    case "assistant_text_delta": return { type: "assistant_text_delta", text: text(data["text"]) };
    case "done": {
      const metrics = record(data["metrics"]);
      const complete = boolean(metrics["completed"]);
      const thinker = boolean(metrics["thinker_eos"]);
      const talker = metrics["talker_eos"] === null ? null : boolean(metrics["talker_eos"]);
      if (complete !== (thinker && talker !== false)) throw new Error("Inconsistent Qwen completion evidence");
      const duration = finite(metrics["output_seconds"]);
      const rtf = nullableNumber(metrics["rtf"]);
      const first = nullableNumber(metrics["first_pcm_seconds"]);
      if (duration < 0 || (rtf !== null && rtf < 0) || (first !== null && first < 0)) throw new Error("Invalid Qwen timings");
      if ((duration === 0 ? rtf !== null || first !== null : rtf === null || first === null)) throw new Error("Qwen speech timings require output audio");
      return { type: "done", complete, value: { basis: "output", rtf, audioSeconds: duration, latencyMs: first === null ? null : first * 1000, latencyLabel: "First PCM", backlogSeconds: null } };
    }
    case "error": return { type: "error", message: text(data["message"]) };
    default: throw new Error("Unknown Qwen session event");
  }
}
