import type { AddedSynthesisId } from "../frontend/domain/experiment.js";
import { expect, test } from "bun:test";
import { decode as nemotron } from "../frontend/models/nemotron/protocol.js";
import { decode as sesame } from "../frontend/models/synthesis/protocol.js";
import { decodeCatalog } from "../frontend/transport/catalog.js";

test("model catalog validates identity, workflow and sample rates together", () => {
  const model = { id: "sesame", kind: "synthesis", endpoint: "/api/sesame/session", output_sample_rate: 24000,
    label: "Sesame", description: "Speech", backend: "PyTorch", available: true, reason: null };
  expect(decodeCatalog([model])[0]?.id).toBe("sesame");
  expect(() => decodeCatalog([{ ...model, output_sample_rate: 16000 }])).toThrow();
  expect(() => decodeCatalog([model, model])).toThrow();
  expect(() => decodeCatalog([{ ...model, endpoint: "https://elsewhere.invalid" }])).toThrow();
  expect(() => decodeCatalog([{ ...model, available: "true" }])).toThrow();
});

test("Nemotron wire events retain the input RTF boundary", () => {
  const event = nemotron(JSON.stringify({ type: "metrics", frames: 3, rtf: .8, p95_ms: 72, input_seconds: .24, backlog_seconds: .08 }));
  expect(event.type).toBe("metrics");
  if (event.type !== "metrics") throw new Error("Wrong event");
  expect(event.value.basis).toBe("input");
  expect(event.value.latencyLabel).toBe("Frame p95");
  expect(event.value.backlogSeconds).toBe(.08);
  expect(() => nemotron('{"type":"metrics","frames":1,"rtf":"fast"}')).toThrow();
  expect(() => nemotron('{"type":"surprise"}')).toThrow();
  expect(() => nemotron('null')).toThrow();
});

test("Nemotron function-channel text remains observable without executing it", () => {
  expect(nemotron('{"type":"function_delta","text":"lookup_weather"}')).toEqual({
    type: "function_delta", text: "lookup_weather",
  });
  expect(() => nemotron('{"type":"function_delta","text":42}')).toThrow();
});

test("Sesame validates completion and reports output RTF separately", () => {
  const metrics = { eos: true, output_rtf: .9, audio_seconds: 3.2, first_pcm_seconds: .3 };
  const event = sesame(JSON.stringify({ type: "complete", metrics }));
  if (event.type !== "complete") throw new Error("Wrong event");
  expect(event.value.basis).toBe("output");
  expect(event.value.latencyMs).toBe(300);
  expect(event.value.backlogSeconds).toBeNull();
  expect(() => sesame(JSON.stringify({ type: "incomplete", metrics }))).toThrow();
  expect(() => sesame('{"type":"ready","sample_rate":16000}')).toThrow();
  expect(() => sesame(JSON.stringify({ type: "complete", metrics: { ...metrics, output_rtf: -1 } }))).toThrow();
});

test("synthesis preserves unknown completion and validates negotiated sample rates", () => {
  const metrics = { eos: null, completion: "unverified", output_rtf: 0.5, audio_seconds: 2, first_pcm_seconds: 0.2 };
  expect(sesame(JSON.stringify({ type: "unverified", metrics })).type).toBe("unverified");
  expect(() => sesame(JSON.stringify({ type: "complete", metrics }))).toThrow();
  expect(() => sesame(JSON.stringify({ type: "complete", metrics: { ...metrics, eos: false, completion: "complete" } }))).toThrow();
  expect(() => sesame(JSON.stringify({ type: "incomplete", metrics: { ...metrics, eos: true, completion: "incomplete" } }))).toThrow();
  expect(sesame(JSON.stringify({ type: "complete", metrics: { ...metrics, completion: "complete" } })).type).toBe("complete");
  expect(sesame('{"type":"ready","sample_rate":44100}', 44100).type).toBe("ready");
  expect(() => sesame('{"type":"ready","sample_rate":24000}', 44100)).toThrow();
});

test("added catalog workflows reject swapped endpoints and sample rates", () => {
  const models: readonly (readonly [AddedSynthesisId, number])[] = [["phonon", 24000], ["kokoro", 24000], ["audio8", 44100], ["ming_tts", 44100],
    ["drama", 44100], ["maya", 24000], ["omnivoice", 24000], ["kitten", 24000]];
  for (const [id, rate] of models) {
    const row = { id, kind: "synthesis", endpoint: `/api/${id}/session`, output_sample_rate: rate,
      label: String(id), description: "Speech", backend: "Local", available: true, reason: null };
    expect(decodeCatalog([row])[0]?.id).toBe(id);
    expect(() => decodeCatalog([{ ...row, output_sample_rate: 16000 }])).toThrow();
    expect(() => decodeCatalog([{ ...row, endpoint: "/api/sesame/session" }])).toThrow();
  }
});

test("new voice catalogs preserve duplex and turn-based contracts", () => {
  const base = { label: "Voice model", description: "Speech", backend: "Local", available: true, reason: null };
  const qwen = { ...base, id: "qwen_omni", kind: "conversation", endpoint: "/api/qwen_omni/session", output_sample_rate: 24000 };
  const minicpm = { ...base, id: "minicpm", kind: "duplex", endpoint: "/api/minicpm/session", input_sample_rate: 16000, output_sample_rate: 24000 };
  expect(decodeCatalog([qwen, minicpm]).map(model => model.kind)).toEqual(["conversation", "duplex"]);
  expect(() => decodeCatalog([{ ...qwen, kind: "duplex" }])).toThrow();
  expect(() => decodeCatalog([{ ...minicpm, input_sample_rate: 24000 }])).toThrow();
  expect(() => decodeCatalog([{ ...qwen, endpoint: minicpm.endpoint }])).toThrow();
});
