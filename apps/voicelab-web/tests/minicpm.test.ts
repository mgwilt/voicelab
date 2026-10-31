import { describe, expect, test } from "bun:test";
import { decode } from "../frontend/models/minicpm/protocol.js";
import { FrameBuffer } from "../frontend/models/minicpm/session.js";
describe("MiniCPM protocol", () => {
  test("checks negotiated audio rates", () => {
    expect(() => decode(JSON.stringify({ type: "ready", input_sample_rate: 24000, output_sample_rate: 24000 }), "full_duplex")).toThrow();
  });
  test("keeps caps incomplete and switches RTF basis", () => {
    const data = JSON.stringify({ type: "done", metrics: { completed: false, input_seconds: 3, output_seconds: 2, rtf: 0.5, first_pcm_seconds: 0.1 } });
    const turn = decode(data, "turn_based");
    expect(turn.type === "done" && turn.complete).toBe(false);
    expect(turn.type === "done" && turn.value.basis).toBe("output");
    const live = decode(data, "full_duplex");
    expect(live.type === "done" && live.value.basis).toBe("input");
  });
  test("assembles arbitrary PCM chunks and pads only the final frame", () => {
    const buffer = new FrameBuffer(); const frames: ArrayBuffer[] = [];
    buffer.push(new Uint8Array(14000).fill(1).buffer, frame => frames.push(frame));
    buffer.push(new Uint8Array(20000).fill(2).buffer, frame => frames.push(frame));
    expect(frames.map(frame => frame.byteLength)).toEqual([32000]);
    buffer.finish(frame => frames.push(frame));
    expect(frames.map(frame => frame.byteLength)).toEqual([32000, 32000]);
    expect(new Uint8Array(frames[1] ?? new ArrayBuffer(0))[1999]).toBe(2);
    expect(new Uint8Array(frames[1] ?? new ArrayBuffer(0))[2000]).toBe(0);
    buffer.finish(frame => frames.push(frame)); expect(frames.length).toBe(2);
  });
});
