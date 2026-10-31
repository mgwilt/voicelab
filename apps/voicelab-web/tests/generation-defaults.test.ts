import { expect, test } from "bun:test";
import { createPanel as audio8 } from "../frontend/models/audio8/panel.js";
import { createPanel as drama } from "../frontend/models/drama/panel.js";
import { MiniCPMPanel } from "../frontend/models/minicpm/panel.js";

const base = { label: "Test", description: "Test", backend: "Test", available: true, reason: null } as const;

test("Audio8 sends the author's sampled baseline from untouched controls", async () => {
  const panel = audio8({ ...base, kind: "synthesis", id: "audio8", endpoint: "/api/audio8/session", outputSampleRate: 44100 });
  try {
    expect(await panel.conversationConfig?.()).toMatchObject({ temperature: 0.8, top_p: 0.95, top_k: 50, do_sample: true });
  } finally { panel.dispose(); }
});

test("Drama starts at top-k 30 and still permits runtime defaults or explicit overrides", async () => {
  const panel = drama({ ...base, kind: "synthesis", id: "drama", endpoint: "/api/drama/session", outputSampleRate: 44100 });
  try {
    expect(await panel.conversationConfig?.()).toMatchObject({ temperature: 1, top_p: 0.9, top_k: 30 });
    const label = [...panel.element.querySelectorAll("label")].find(label => label.textContent?.startsWith("Top k"));
    const input = label?.querySelector("input");
    if (!(input instanceof HTMLInputElement)) throw new Error("Missing top-k control");
    input.value = "";
    expect(await panel.conversationConfig?.()).toMatchObject({ top_k: null });
    input.value = "81";
    expect(await panel.conversationConfig?.()).toMatchObject({ top_k: 81 });
  } finally { panel.dispose(); }
});

test("MiniCPM sends specialized voice settings rather than generic checkpoint sampling", async () => {
  const panel = new MiniCPMPanel({ ...base, kind: "duplex", id: "minicpm", endpoint: "/api/minicpm/session", inputSampleRate: 16000, outputSampleRate: 24000 });
  try {
    expect(await panel.conversationConfig()).toMatchObject({ temperature: 0.7, top_p: 0.8, top_k: 100, tts_temperature: 0.8, length_penalty: 1.1 });
  } finally { panel.dispose(); }
});
