import type { Measurement } from "../../domain/experiment.js";
import { boolean, finite, parse, record, text } from "../../transport/decode.js";

export type ToolMessage =
  | { readonly type: "loading" | "ready" }
  | { readonly type: "assistant_text_delta" | "function_delta"; readonly text: string }
  | { readonly type: "done"; readonly complete: boolean; readonly value: Measurement }
  | { readonly type: "error"; readonly message: string };

export function decodeTool(raw: string): ToolMessage {
  const data = parse(raw);
  switch (data["type"]) {
    case "loading": return { type: "loading" };
    case "ready":
      if (data["input_sample_rate"] !== 16000 || data["output_sample_rate"] !== 22050) throw new Error("Unexpected Nemotron tool-test sample rates");
      return { type: "ready" };
    case "assistant_text_delta": case "function_delta": return { type: data["type"], text: text(data["text"]) };
    case "done": {
      const metrics = record(data["metrics"]);
      return { type: "done", complete: boolean(metrics["completed"]), value: {
        basis: "input", rtf: finite(metrics["rtf"]), audioSeconds: finite(metrics["input_seconds"]),
        latencyLabel: "First PCM", latencyMs: null, backlogSeconds: null,
      } };
    }
    case "error": return { type: "error", message: text(data["message"]) };
    default: throw new Error("Unknown Nemotron tool-test event");
  }
}
