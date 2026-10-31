import { expect, test } from "bun:test";
import type { ControlPanel } from "../frontend/components/panel.js";
import { ConversationView } from "../frontend/conversation/view.js";
import type { ConversationConnection } from "../frontend/conversation/client.js";
import type { Model } from "../frontend/domain/session.js";

const model: Model = { conversationModes: ["cascade"], id: "kokoro", kind: "synthesis", label: "Kokoro", description: "Voice", backend: "MLX", available: true, reason: null, endpoint: "/api/kokoro/session", outputSampleRate: 24000 };
const tick = async (): Promise<void> => { await new Promise(resolve => setTimeout(resolve, 0)); };
function button(view: ConversationView, label: string): HTMLButtonElement {
  const result = [...view.element.querySelectorAll("button")].find(node => node.textContent === label);
  if (result === undefined) throw new Error(`Missing ${label}`);
  return result;
}
function panel(read: () => Promise<object>): ControlPanel {
  return { element: document.createElement("div"), conversationConfig: read, setDisabled: () => {}, dispose: () => {}, createSession: () => { throw new Error("Direct run must not be used"); } };
}

test("ending during async settings prevents a late connection", async () => {
  let finish: ((settings: object) => void) | undefined;
  const settings = new Promise<object>(resolve => { finish = resolve; });
  let calls = 0;
  const view = new ConversationView([model], () => panel(() => settings), () => { calls++; throw new Error("Must not connect"); });
  await tick(); button(view, "Start conversation").click(); await tick();
  await view.stop(); finish?.({ text: "Hello" }); await tick();
  expect(calls).toBe(0); expect(button(view, "Start conversation").disabled).toBe(false);
  await view.dispose();
});

test("start, mute, interrupt, and dispose share one connection lifetime", async () => {
  let starts = 0; let stops = 0; let muted = false; let interrupted = false;
  const connection: ConversationConnection = {
    audio: document.createElement("audio"), start: async () => { starts++; },
    mute: value => { muted = value; }, interrupt: async () => { interrupted = true; }, stop: async () => { stops++; },
  };
  const view = new ConversationView([model], () => panel(async () => ({ text: "Hello" })), () => connection);
  await tick(); button(view, "Start conversation").click(); await tick();
  expect(starts).toBe(1); expect(button(view, "Start conversation").disabled).toBe(true);
  button(view, "Mute microphone").click(); button(view, "Interrupt").click(); await tick();
  expect(muted).toBe(true); expect(interrupted).toBe(true);
  await view.dispose(); expect(stops).toBe(1);
});

test("speaker mute leaves microphone and transcript delivery active", async () => {
  let microphoneChanges = 0; let stopped = 0;
  let receive: ((event: import("../frontend/conversation/events.js").Notice) => void) | undefined;
  const audio = document.createElement("audio");
  const view = new ConversationView([model], () => panel(async () => ({})), event => {
    receive = event;
    return { audio, start: async () => {}, mute: () => { microphoneChanges++; }, interrupt: async () => {}, stop: async () => { stopped++; } };
  });
  await tick(); button(view, "Start conversation").click(); await tick();
  button(view, "Mute speaker").click();
  expect(audio.muted).toBe(true);
  expect(microphoneChanges).toBe(0); expect(stopped).toBe(0);
  receive?.({ type: "text", text: "Still listening", textMode: "snapshot", speaker: "assistant", final: false, turn: 0, stage: "", seconds: null, outputRtf: null, inputRtf: null });
  expect(view.element.querySelector(".conversation-message.assistant p")?.textContent).toBe("Still listening");
  button(view, "Mute microphone").click();
  expect(microphoneChanges).toBe(1); expect(audio.muted).toBe(true);
  await view.stop();
  expect(button(view, "Mute microphone").getAttribute("aria-pressed")).toBe("false");
  button(view, "Start conversation").click(); await tick();
  expect(audio.muted).toBe(true);
  button(view, "Unmute speaker").click();
  expect(audio.muted).toBe(false); expect(microphoneChanges).toBe(1);
  await view.dispose();
});

test("repeated calls reuse one media owner and retain the workbench components", async () => {
  let owners = 0; let starts = 0;
  const view = new ConversationView([model], () => panel(async () => ({})), () => {
    owners++;
    return { audio: document.createElement("audio"), start: async () => { starts++; }, mute: () => {}, interrupt: async () => {}, stop: async () => {} };
  });
  await tick();
  expect(view.element.querySelector(".model-explorer")).not.toBeNull();
  expect(view.element.querySelector(".workbench-tabs")).not.toBeNull();
  expect(view.element.querySelector(".runbar")).not.toBeNull();
  button(view, "Start conversation").click(); await tick();
  await view.stop(); button(view, "Start conversation").click(); await tick();
  expect(owners).toBe(1); expect(starts).toBe(2);
  await view.dispose();
});

test("echo warnings remain visible when ordinary call status changes", async () => {
  const view = new ConversationView([model], () => panel(async () => ({})), (_event, status, _ended, processing) => ({
    audio: document.createElement("audio"),
    start: async () => {
      processing({ callAudioRouting: "unavailable", echoCancellation: false, noiseSuppression: true, autoGainControl: true, configurationFailed: false });
      status("Listening");
    },
    mute: () => {}, interrupt: async () => {}, stop: async () => {},
  }));
  await tick(); button(view, "Start conversation").click(); await tick();
  expect(view.element.querySelector(".session-status")?.textContent).toBe("Listening");
  expect(view.element.querySelector(".microphone-status")?.textContent).toContain("echo cancellation is off");
  await view.dispose();
});

test("cumulative Nemotron updates replace partial text without multiplying prefixes", async () => {
  let receive: ((event: import("../frontend/conversation/events.js").Notice) => void) | undefined;
  const view = new ConversationView([model], () => panel(async () => ({})), event => {
    receive = event;
    return { audio: document.createElement("audio"), start: async () => {}, mute: () => {}, interrupt: async () => {}, stop: async () => {} };
  });
  await tick(); button(view, "Start conversation").click(); await tick();
  const base = { type: "text", final: false, turn: 0, stage: "", seconds: null, outputRtf: null, inputRtf: null } as const;
  for (const text of ["I", "I am", "I am a language model", "I am a language model."]) {
    receive?.({ ...base, speaker: "assistant", textMode: "snapshot", text });
  }
  receive?.({ ...base, speaker: "user", textMode: "snapshot", text: "Who" });
  receive?.({ ...base, speaker: "user", textMode: "snapshot", text: "Who are you?" });
  expect(view.element.querySelector(".conversation-message.assistant p")?.textContent).toBe("I am a language model.");
  expect(view.element.querySelector(".conversation-message.user p")?.textContent).toBe("Who are you?");
  // Tokenizers may revise an earlier partial byte sequence, not just grow the prefix.
  receive?.({ ...base, speaker: "assistant", textMode: "snapshot", text: "Hello, caf�" });
  receive?.({ ...base, speaker: "assistant", textMode: "snapshot", text: "Hello, café." });
  expect(view.element.querySelector(".conversation-message.assistant p")?.textContent).toBe("Hello, café.");
  // Ordinary deltas must retain genuine repetitions, including identical chunks.
  for (const text of ["very ", "very ", "good"]) receive?.({ ...base, turn: 1, speaker: "assistant", textMode: "delta", text });
  const messages = view.element.querySelectorAll(".conversation-message.assistant p");
  expect(messages[1]?.textContent).toBe("very very good");
  receive?.({ ...base, turn: 1, final: true, speaker: "assistant", textMode: "delta", text: "Very, very good." });
  expect(messages[1]?.textContent).toBe("Very, very good.");
  await view.dispose();
});
