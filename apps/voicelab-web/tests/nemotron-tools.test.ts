import { describe, expect, test } from "bun:test";
import { decodeTool } from "../frontend/models/nemotron/tool_protocol.js";

describe("offline Nemotron supplied-response protocol", () => {
  test("labels both-pass RTF as input and preserves incomplete outcome", () => {
    const message = decodeTool(JSON.stringify({ type: "done", metrics: {
      rtf: 1.7, input_seconds: 21.8, completed: false,
    } }));
    expect(message.type).toBe("done");
    if (message.type !== "done") throw new Error("Expected done");
    expect(message.complete).toBe(false);
    expect(message.value.basis).toBe("input");
    expect(message.value.latencyMs).toBeNull();
  });
  test("validates rates, metric types, and tool-call text", () => {
    expect(() => decodeTool('{"type":"ready","input_sample_rate":16000,"output_sample_rate":24000}')).toThrow();
    expect(() => decodeTool('{"type":"done","metrics":{"rtf":"1.7","input_seconds":20,"completed":true}}')).toThrow();
    expect(decodeTool('{"type":"function_delta","text":"lookup"}')).toEqual({ type: "function_delta", text: "lookup" });
  });
});

test("Nemotron workflow disables hidden recording controls and releases listeners", async () => {
  const { DuplexPanel } = await import("../frontend/models/nemotron/panel.js");
  const panel = new DuplexPanel({ id: "nemotron", kind: "duplex", label: "Nemotron", description: "Test",
    backend: "MLX", available: true, reason: null, endpoint: "/api/nemotron/session", inputSampleRate: 16000, outputSampleRate: 22050 });
  const mode = panel.element.querySelector("select");
  const file = panel.element.querySelector<HTMLInputElement>('input[type="file"]');
  if (mode === null || file === null) throw new Error("Missing tool workflow controls");
  expect(file.disabled).toBe(true);
  mode.value = "tools"; mode.dispatchEvent(new Event("change"));
  expect(file.disabled).toBe(false);
  panel.setDisabled(true); expect(file.disabled).toBe(true);
  panel.setDisabled(false); expect(file.disabled).toBe(false);
  mode.value = "live"; mode.dispatchEvent(new Event("change"));
  expect(file.disabled).toBe(true);
  panel.dispose();
  mode.value = "tools"; mode.dispatchEvent(new Event("change"));
  expect(file.disabled).toBe(true);
});
