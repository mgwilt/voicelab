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
    if (id === "phonon" && row["kind"] === "synthesis" &&
        row["endpoint"] === "/api/phonon/session" && row["output_sample_rate"] === 24000) {
      return { ...base, id, kind: "synthesis", endpoint: "/api/phonon/session", outputSampleRate: 24000 };
    }
    if (id === "kokoro" && row["kind"] === "synthesis" &&
        row["endpoint"] === "/api/kokoro/session" && row["output_sample_rate"] === 24000) {
      return { ...base, id, kind: "synthesis", endpoint: "/api/kokoro/session", outputSampleRate: 24000 };
    }
    if (id === "audio8" && row["kind"] === "synthesis" &&
        row["endpoint"] === "/api/audio8/session" && row["output_sample_rate"] === 44100) {
      return { ...base, id, kind: "synthesis", endpoint: "/api/audio8/session", outputSampleRate: 44100 };
    }
    if (id === "ming_tts" && row["kind"] === "synthesis" &&
        row["endpoint"] === "/api/ming_tts/session" && row["output_sample_rate"] === 44100) {
      return { ...base, id, kind: "synthesis", endpoint: "/api/ming_tts/session", outputSampleRate: 44100 };
    }
    if (id === "drama" && row["kind"] === "synthesis" &&
        row["endpoint"] === "/api/drama/session" && row["output_sample_rate"] === 44100) {
      return { ...base, id, kind: "synthesis", endpoint: "/api/drama/session", outputSampleRate: 44100 };
    }
    if (id === "maya" && row["kind"] === "synthesis" &&
        row["endpoint"] === "/api/maya/session" && row["output_sample_rate"] === 24000) {
      return { ...base, id, kind: "synthesis", endpoint: "/api/maya/session", outputSampleRate: 24000 };
    }
    if (id === "omnivoice" && row["kind"] === "synthesis" &&
        row["endpoint"] === "/api/omnivoice/session" && row["output_sample_rate"] === 24000) {
      return { ...base, id, kind: "synthesis", endpoint: "/api/omnivoice/session", outputSampleRate: 24000 };
    }
    if (id === "kitten" && row["kind"] === "synthesis" &&
        row["endpoint"] === "/api/kitten/session" && row["output_sample_rate"] === 24000) {
      return { ...base, id, kind: "synthesis", endpoint: "/api/kitten/session", outputSampleRate: 24000 };
    }
    if (id === "personaplex" && row["kind"] === "duplex" &&
        row["endpoint"] === "/api/personaplex/session" && row["input_sample_rate"] === 24000 && row["output_sample_rate"] === 24000) {
      return { ...base, id, kind: "duplex", endpoint: "/api/personaplex/session", inputSampleRate: 24000, outputSampleRate: 24000 };
    }
    if (id === "ming_audio" && row["kind"] === "conversation" &&
        row["endpoint"] === "/api/ming_audio/session" && row["output_sample_rate"] === 44100) {
      return { ...base, id, kind: "conversation", endpoint: "/api/ming_audio/session", outputSampleRate: 44100 };
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
