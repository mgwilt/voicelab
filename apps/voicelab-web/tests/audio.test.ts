import { expect, test } from "bun:test";
import { acquireStream } from "../frontend/audio/permission.js";
import { PcmResampler } from "../frontend/audio/resampler.js";

function capture(block: number, rate = 48000): Uint8Array {
  const frames: ArrayBuffer[] = [];
  const resampler = new PcmResampler(rate);
  const samples = Float32Array.from({ length: 30720 }, (_, i) => Math.sin(i * .07) * .2);
  for (let i = 0; i < samples.length; i += block) resampler.push(samples.slice(i, i + block), frame => frames.push(frame));
  expect(frames.every(frame => frame.byteLength === 2560)).toBe(true);
  return new Uint8Array(Buffer.concat(frames.map(frame => Buffer.from(frame))));
}

test("resampler preserves its timeline across device block boundaries", () => {
  const frames = capture(128);
  expect(frames.byteLength).toBe(8 * 2560);
  expect(frames).toEqual(capture(256));
  expect(capture(128, 44100)).toEqual(capture(512, 44100));
  expect(() => new PcmResampler(0)).toThrow();
});

test("a microphone permission grant after cancellation releases all tracks", async () => {
  let grant: ((stream: { getTracks(): { stop(): void }[] }) => void) | undefined;
  let stopped = 0;
  const pending = new Promise<{ getTracks(): { stop(): void }[] }>(resolve => { grant = resolve; });
  const cancellation = new AbortController();
  const acquisition = acquireStream(pending, cancellation.signal);
  cancellation.abort();
  await expect(acquisition).rejects.toThrow("Microphone cancelled");
  if (grant === undefined) throw new Error("Missing deferred resolver");
  grant({ getTracks: () => [{ stop: () => stopped++ }, { stop: () => stopped++ }] });
  await Promise.resolve();
  expect(stopped).toBe(2);
});

test("PersonaPlex resampling keeps continuous 24 kHz frames", () => {
  const source = new Float32Array(48000).fill(0.5);
  const resampler = new PcmResampler(48000, 24000);
  const frames: ArrayBuffer[] = [];
  for (let i = 0; i < source.length; i += 128) resampler.push(source.subarray(i, i + 128), frame => frames.push(frame));
  expect(frames.length).toBe(12);
  expect(frames.every(frame => frame.byteLength === 3840)).toBe(true);
  const last = frames.at(-1);
  if (last === undefined) throw new Error("Missing audio frame");
  expect(new DataView(last).getInt16(3838, true)).toBe(16384);
});
