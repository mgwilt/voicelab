import type { Measurement } from "../../domain/experiment.js";
import { finite, nullableNumber, parse, record, text } from "../../transport/decode.js";

export interface StartRequest { readonly type: "start"; readonly prompt: string; readonly seed: number }
export type Message =
  | { readonly type: "loading" }
  | { readonly type: "ready"; readonly setupSeconds: number }
  | { readonly type: "metrics" | "done"; readonly value: Measurement }
  | { readonly type: "assistant_text_delta" | "user_transcript_delta" | "function_delta"; readonly text: string }
  | { readonly type: "error"; readonly message: string };

function measurement(value: unknown): Measurement {
  const data = record(value);
  const frames = finite(data["frames"]);
  return {
    basis: "input", rtf: nullableNumber(data["rtf"]), latencyLabel: "Frame p95",
    latencyMs: nullableNumber(data["p95_ms"]),
    audioSeconds: frames === 0 ? 0 : finite(data["input_seconds"]),
    backlogSeconds: data["backlog_seconds"] === undefined ? null : finite(data["backlog_seconds"]),
  };
}
export function decode(raw: string): Message {
  const data = parse(raw);
  const type = data["type"];
  switch (type) {
    case "loading": return { type };
    case "ready": return { type, setupSeconds: finite(data["setup_seconds"]) };
    case "metrics": return { type, value: measurement(data) };
    case "done": return { type, value: measurement(data["metrics"]) };
    case "assistant_text_delta": case "user_transcript_delta": case "function_delta":
      return { type, text: text(data["text"]) };
    case "error": return { type, message: text(data["message"]) };
    default: throw new Error(`Unknown Nemotron event: ${String(type)}`);
  }
}
