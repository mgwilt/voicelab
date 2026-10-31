import { describe, expect, test } from "bun:test";
import { decodeCreated, decodeNotice, errorMessage } from "../frontend/conversation/events.js";

describe("conversation wire boundaries", () => {
  const valid = { type: "metrics", text: "", speaker: "assistant", final: false, turn: 1, stage: "tts", seconds: 1.2, output_rtf: 0.4, input_rtf: null };
  test("keeps input and output RTF distinct", () => {
    expect(decodeNotice(valid).outputRtf).toBe(0.4);
    expect(decodeNotice(valid).inputRtf).toBeNull();
  });
  test("rejects malformed and non-finite events", () => {
    for (const value of [null, {}, { ...valid, type: "unknown" }, { ...valid, seconds: NaN }, { ...valid, output_rtf: -1 }, { ...valid, turn: 1.2 }, { ...valid, final: "yes" }]) expect(() => decodeNotice(value)).toThrow();
  });
  test("signaling endpoints stay bound to the created session and origin", () => {
    expect(decodeCreated({ session_id: "abc", offer_url: "/api/conversations/abc/offer" }).sessionId).toBe("abc");
    for (const offer_url of ["https://example.com/offer", "/api/conversations/other/offer", "//example.com/offer"]) expect(() => decodeCreated({ session_id: "abc", offer_url })).toThrow();
  });
});

test("API errors show readable messages without exposing wire JSON", () => {
  expect(errorMessage({ detail: [{ msg: "Enter a system prompt" }] }, 422)).toBe("Enter a system prompt");
  expect(errorMessage({ detail: "Another model session is active" }, 409)).toBe("Another model session is active");
  expect(errorMessage(null, 500)).toContain("Try again");
});
