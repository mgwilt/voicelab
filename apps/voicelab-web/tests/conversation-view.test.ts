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
  button(view, "Mute").click(); button(view, "Interrupt").click(); await tick();
  expect(muted).toBe(true); expect(interrupted).toBe(true);
  await view.dispose(); expect(stops).toBe(1);
});
