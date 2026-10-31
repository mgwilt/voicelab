import { expect, test } from "bun:test";
import { decode } from "../frontend/models/qwen_omni/protocol.js";

const base = { completed: true, thinker_eos: true, talker_eos: true,
  output_seconds: 2, rtf: .7, first_pcm_seconds: .2 };
const event = (metrics: object): string => JSON.stringify({ type: "done", metrics });

test("Qwen requires both decoders to complete speech", () => {
  expect(decode(event(base)).type).toBe("done");
  expect(() => decode(event({ ...base, talker_eos: false }))).toThrow();
  expect(() => decode(event({ ...base, thinker_eos: false }))).toThrow();
  expect(decode(event({ ...base, completed: false, talker_eos: false })).type).toBe("done");
});

test("Qwen speech timings are present together, text-only has neither", () => {
  expect(() => decode(event({ ...base, rtf: null }))).toThrow();
  expect(() => decode(event({ ...base, first_pcm_seconds: null }))).toThrow();
  expect(() => decode(event({ ...base, output_seconds: 0 }))).toThrow();
  expect(decode(event({ ...base, output_seconds: 0, rtf: null, first_pcm_seconds: null, talker_eos: null })).type).toBe("done");
  expect(() => decode('{"type":"ready","input_sample_rate":16000,"output_sample_rate":16000}')).toThrow();
});
