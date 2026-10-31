import type { Model } from "../domain/experiment.js";
import { boolean, record, text } from "./decode.js";

export function decodeCatalog(value: unknown): readonly Model[] {
  if (!Array.isArray(value)) throw new Error("Invalid model catalog");
  const ids = new Set<string>();
  return value.map((entry: unknown): Model => {
    const row = record(entry);
    const id = text(row["id"]);
    if (ids.has(id)) throw new Error("Duplicate model in catalog");
    ids.add(id);
    const base = {
      label: text(row["label"]), description: text(row["description"]),
      backend: text(row["backend"]), available: boolean(row["available"]),
      reason: row["reason"] === null ? null : text(row["reason"]),
    };
    if (id === "nemotron" && row["kind"] === "duplex" &&
        row["endpoint"] === "/api/nemotron/session" &&
        row["input_sample_rate"] === 16000 && row["output_sample_rate"] === 22050) {
      return { ...base, id, kind: "duplex", endpoint: "/api/nemotron/session",
        inputSampleRate: 16000, outputSampleRate: 22050 };
    }
    if (id === "sesame" && row["kind"] === "synthesis" &&
        row["endpoint"] === "/api/sesame/session" && row["output_sample_rate"] === 24000) {
      return { ...base, id, kind: "synthesis", endpoint: "/api/sesame/session", outputSampleRate: 24000 };
    }
    if (id === "breeze" && row["kind"] === "synthesis" &&
        row["endpoint"] === "/api/breeze/session" && row["output_sample_rate"] === 24000) {
      return { ...base, id, kind: "synthesis", endpoint: "/api/breeze/session", outputSampleRate: 24000 };
    }
    throw new Error(`Unsupported model contract: ${id}`);
  });
}
export async function loadCatalog(signal: AbortSignal): Promise<readonly Model[]> {
  const response = await fetch("/api/models", { signal, cache: "no-store" });
  if (!response.ok) throw new Error(`Could not load models (${response.status})`);
  const value: unknown = await response.json();
  return decodeCatalog(value);
}
