import { microphone } from "./microphone-fixture.js";
import { expect, spyOn, test } from "bun:test";
import type { BotReadyData, RTVIEventCallbacks } from "@pipecat-ai/client-js";
import { ConversationClient } from "../frontend/conversation/client.js";
import type { Model } from "../frontend/domain/session.js";

const model: Model = { id: "kokoro", kind: "synthesis", label: "Kokoro", description: "Voice", backend: "MLX", available: true, reason: null, endpoint: "/api/kokoro/session", outputSampleRate: 24000 };


test("hang up, ignore a stale shared-media event, then start another call", async () => {
  let callbacks: RTVIEventCallbacks = {};
  let local = microphone(); let connects = 0; let deleted = 0;
  const fetch = spyOn(globalThis, "fetch").mockImplementation(Object.assign(async (_input: RequestInfo | URL, init?: RequestInit) => {
    if (init?.method === "DELETE") { deleted++; return new Response(null, { status: 204 }); }
    return Response.json({ session_id: "call", offer_url: "/api/conversations/call/offer" });
  }, { preconnect: globalThis.fetch.preconnect }));
  const client = new ConversationClient(() => {}, () => {}, () => {}, options => {
    callbacks = options;
    return {
      initDevices: async () => { local = microphone(); },
      connect: async () => { connects++; return { version: "1.0.0" }; }, disconnect: async () => {}, enableMic: () => {},
      tracks: () => ({ local: { audio: local }, bot: {} }),
    };
  });
  try {
    await client.start(model, "cascade", {}, "Speak briefly");
    await client.stop(); expect(local.readyState).toBe("ended");
    const next = microphone(); callbacks.onTrackStarted?.(next, { id: "local", name: "You", local: true });
    expect(next.readyState).toBe("live");
    await client.start(model, "cascade", {}, "Speak briefly");
    expect(local.readyState).toBe("live"); expect(connects).toBe(2);
    await client.stop(); expect(deleted).toBe(2);
  } finally { fetch.mockRestore(); }
});

test("hang up while waiting for bot-ready does not wait on the SDK connect promise", async () => {
  const fetch = spyOn(globalThis, "fetch").mockImplementation(Object.assign(async (_input: RequestInfo | URL, init?: RequestInit) => init?.method === "DELETE" ? new Response(null, { status: 204 }) : Response.json({ session_id: "call", offer_url: "/api/conversations/call/offer" }), { preconnect: globalThis.fetch.preconnect }));
  let connecting: (() => void) | undefined;
  const entered = new Promise<void>(resolve => { connecting = resolve; });
  const client = new ConversationClient(() => {}, () => {}, () => {}, () => ({
    initDevices: async () => {}, connect: () => { connecting?.(); return new Promise<BotReadyData>(() => {}); },
    disconnect: async () => {}, enableMic: () => {}, tracks: () => ({ local: { audio: microphone() }, bot: {} }),
  }));
  try {
    const starting = client.start(model, "cascade", {}, "");
    await entered; await client.stop(); await starting;
  } finally { fetch.mockRestore(); }
});

test("late microphone permission is released before hangup completes", async () => {
  let granted: (() => void) | undefined;
  const permission = new Promise<void>(resolve => { granted = resolve; });
  const local = microphone(); let closed = false;
  const client = new ConversationClient(() => {}, () => {}, () => { closed = true; }, () => ({
    initDevices: () => permission,
    connect: async () => { throw new Error("Cancelled calls must not connect"); },
    disconnect: async () => {}, enableMic: () => {}, tracks: () => ({ local: { audio: local }, bot: {} }),
  }));
  const starting = client.start(model, "cascade", {}, "");
  const stopping = client.stop();
  await Promise.resolve(); expect(closed).toBe(false);
  granted?.(); await stopping; await starting;
  expect(local.readyState).toBe("ended"); expect(closed).toBe(true);
});

test("capture settings finish before signaling, and replacement tracks are configured", async () => {
  let callbacks: RTVIEventCallbacks = {};
  let configured = false; let replacementConfigured = false;
  let local = microphone({ echoCancellation: true }, async () => { configured = true; });
  const reports: (import("../frontend/audio/processing.js").MicrophoneProcessingReport | null)[] = [];
  const fetch = spyOn(globalThis, "fetch").mockImplementation(Object.assign(async (_input: RequestInfo | URL, init?: RequestInit) => {
    if (init?.method === "DELETE") return new Response(null, { status: 204 });
    expect(configured).toBe(true);
    return Response.json({ session_id: "call", offer_url: "/api/conversations/call/offer" });
  }, { preconnect: globalThis.fetch.preconnect }));
  const client = new ConversationClient(() => {}, () => {}, () => {}, options => {
    callbacks = options;
    return { initDevices: async () => {}, connect: async () => ({ version: "1.0.0" }), disconnect: async () => {}, enableMic: () => {}, tracks: () => ({ local: { audio: local }, bot: {} }) };
  }, report => reports.push(report));
  try {
    await client.start(model, "cascade", {}, "");
    expect(reports.at(-1)?.echoCancellation).toBe(true);
    local = microphone({ echoCancellation: false }, async () => { replacementConfigured = true; });
    callbacks.onTrackStarted?.(local, { id: "local", name: "You", local: true });
    await Promise.resolve();
    expect(replacementConfigured).toBe(true);
    expect(reports.at(-1)?.echoCancellation).toBe(false);
    await client.stop();
  } finally { fetch.mockRestore(); }
});

test("ending during constraint application cannot reconnect an old call after a new start", async () => {
  let finish: (() => void) | undefined; let applying: (() => void) | undefined;
  const pending = new Promise<void>(resolve => { finish = resolve; });
  const entered = new Promise<void>(resolve => { applying = resolve; });
  let local = microphone({}, () => { applying?.(); return pending; });
  const first = local;
  let reservations = 0; let connects = 0;
  const fetch = spyOn(globalThis, "fetch").mockImplementation(Object.assign(async (_input: RequestInfo | URL, init?: RequestInit) => {
    if (init?.method === "DELETE") return new Response(null, { status: 204 });
    reservations++;
    return Response.json({ session_id: "call", offer_url: "/api/conversations/call/offer" });
  }, { preconnect: globalThis.fetch.preconnect }));
  const client = new ConversationClient(() => {}, () => {}, () => {}, () => ({
    initDevices: async () => {}, connect: async () => { connects++; return { version: "1.0.0" }; },
    disconnect: async () => {}, enableMic: () => {}, tracks: () => ({ local: { audio: local }, bot: {} }),
  }));
  try {
    const oldStart = client.start(model, "cascade", {}, "");
    await entered; await client.stop();
    expect(first.readyState).toBe("ended"); expect(reservations).toBe(0);
    local = microphone({ echoCancellation: true });
    await client.start(model, "cascade", {}, "");
    finish?.(); await oldStart;
    expect(reservations).toBe(1); expect(connects).toBe(1);
    await client.stop();
  } finally { fetch.mockRestore(); }
});

test("call audio routing starts before capture and ends after microphone cleanup", async () => {
  const original = Object.getOwnPropertyDescriptor(navigator, "audioSession");
  const audioSession = { type: "auto" };
  Object.defineProperty(navigator, "audioSession", { configurable: true, value: audioSession });
  let local = microphone();
  const fetch = spyOn(globalThis, "fetch").mockImplementation(Object.assign(async (_input: RequestInfo | URL, init?: RequestInit) => init?.method === "DELETE" ? new Response(null, { status: 204 }) : Response.json({ session_id: "call", offer_url: "/api/conversations/call/offer" }), { preconnect: globalThis.fetch.preconnect }));
  const reports: (import("../frontend/audio/processing.js").MicrophoneProcessingReport | null)[] = [];
  const client = new ConversationClient(() => {}, () => {}, () => {}, () => ({
    initDevices: async () => { expect(audioSession.type).toBe("play-and-record"); local = microphone(); },
    connect: async () => ({ version: "1.0.0" }),
    disconnect: async () => { expect(audioSession.type).toBe("play-and-record"); },
    enableMic: () => {}, tracks: () => ({ local: { audio: local }, bot: {} }),
  }), report => reports.push(report));
  try {
    for (let call = 0; call < 2; call++) {
      await client.start(model, "cascade", {}, "");
      expect(reports.at(-1)?.callAudioRouting).toBe("play-and-record");
      await client.stop();
      expect(local.readyState).toBe("ended");
      expect(audioSession.type).toBe("auto");
    }
  } finally {
    fetch.mockRestore();
    if (original === undefined) Reflect.deleteProperty(navigator, "audioSession");
    else Object.defineProperty(navigator, "audioSession", original);
  }
});

test("failed SDK disconnect still waits for late capture before restoring routing", async () => {
  const original = Object.getOwnPropertyDescriptor(navigator, "audioSession");
  const audioSession = { type: "playback" };
  Object.defineProperty(navigator, "audioSession", { configurable: true, value: audioSession });
  let granted: (() => void) | undefined;
  const permission = new Promise<void>(resolve => { granted = resolve; });
  const local = microphone();
  const client = new ConversationClient(() => {}, () => {}, () => {}, () => ({
    initDevices: () => permission, connect: async () => { throw new Error("Cancelled"); },
    disconnect: async () => { throw new Error("SDK disconnect failed"); },
    enableMic: () => {}, tracks: () => ({ local: { audio: local }, bot: {} }),
  }));
  try {
    const starting = client.start(model, "cascade", {}, "");
    const stopping = client.stop();
    const rejected = stopping.then(() => null, (error: unknown) => error);
    await Promise.resolve();
    expect(audioSession.type).toBe("play-and-record");
    granted?.();
    const failure = await rejected; await starting;
    expect(failure).toBeInstanceOf(Error);
    expect(failure instanceof Error ? failure.message : "").toBe("SDK disconnect failed");
    expect(local.readyState).toBe("ended");
    expect(audioSession.type).toBe("playback");
  } finally {
    if (original === undefined) Reflect.deleteProperty(navigator, "audioSession");
    else Object.defineProperty(navigator, "audioSession", original);
  }
});
