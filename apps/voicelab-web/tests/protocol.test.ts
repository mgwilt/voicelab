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
