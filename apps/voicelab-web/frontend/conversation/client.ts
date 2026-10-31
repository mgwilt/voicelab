import { PipecatClient, type RTVIEventCallbacks } from "@pipecat-ai/client-js";
import { SmallWebRTCTransport } from "@pipecat-ai/small-webrtc-transport";
import { decodeCreated, decodeNotice, errorMessage, type ConversationMode, type Notice } from "./events.js";
import { MicrophoneProcessing, type ReportMicrophoneProcessing } from "../audio/processing.js";
import { CallAudioSession } from "../audio/call-session.js";
import type { Model } from "../domain/session.js";

export interface ConversationConnection {
  readonly audio: HTMLAudioElement;
  start(model: Model, mode: ConversationMode, settings: object, systemPrompt: string): Promise<void>;
  mute(muted: boolean): void;
  interrupt(): Promise<void>;
  stop(): Promise<void>;
}
export type ConnectionFactory = (event: (event: Notice) => void, status: (message: string) => void, ended: () => void, processing: ReportMicrophoneProcessing) => ConversationConnection;

type ClientSDK = Pick<PipecatClient, "connect" | "disconnect" | "enableMic" | "initDevices" | "tracks">;
type SDKFactory = (callbacks: RTVIEventCallbacks) => ClientSDK;

function createBrowserSDK(callbacks: RTVIEventCallbacks): ClientSDK {
  return new PipecatClient({
    transport: new SmallWebRTCTransport({ iceServers: [], waitForICEGathering: true }),
    enableMic: true, enableCam: false, callbacks,
  });
}

/** Reuses one SDK/media manager across calls; each call owns a new reservation. */
export class ConversationClient {
  private readonly sdk: ClientSDK;
  private lifetime = new AbortController();
  private initializing: Promise<void> | null = null;
  private active = false;
  private sessionId: string | null = null;
  private closing: Promise<void> | null = null;
  private processing: MicrophoneProcessing | null = null;
  private audioSession: CallAudioSession | null = null;
  readonly audio = document.createElement("audio");

  constructor(private readonly event: (event: Notice) => void, private readonly status: (message: string) => void, private readonly ended: () => void, createSDK: SDKFactory = createBrowserSDK, private readonly reportProcessing: ReportMicrophoneProcessing = () => {}) {
    this.audio.autoplay = true;
    this.audio.controls = true;
    this.audio.setAttribute("aria-label", "Conversation audio");
    this.sdk = createSDK({
        onServerMessage: (value: unknown) => {
          if (this.lifetime.signal.aborted) return;
          try { this.event(decodeNotice(value)); } catch (error) { this.status(String(error)); void this.stop().catch(error => this.status(String(error))); }
        },
        onTrackStarted: (track, participant) => {
          // The SDK's shared media object can deliver tracks after disconnect.
          // Those tracks may belong to a later call; never stop them here.
          if (!this.active || this.lifetime.signal.aborted) return;
          if (track.kind === "audio" && participant?.local === true) {
            void this.processing?.configure(track);
          } else if (track.kind === "audio") {
            this.audio.srcObject = new MediaStream([track]);
            void this.audio.play().catch(() => this.status("Tap the audio play button to hear the conversation."));
          }
        },
        onDisconnected: () => { if (!this.lifetime.signal.aborted) { this.status("Conversation ended"); void this.stop().catch(error => this.status(String(error))); } },
        onError: () => { if (this.lifetime.signal.aborted) return; this.status("The voice connection failed. End the call and try again."); void this.stop().catch(error => this.status(String(error))); },
    });
  }

  async start(model: Model, mode: ConversationMode, settings: object, systemPrompt: string): Promise<void> {
    if (this.closing !== null) await this.closing;
    if (this.active) throw new Error("A conversation is already active");
    this.lifetime = new AbortController(); this.closing = null; this.active = true;
    const audioSession = new CallAudioSession();
    this.audioSession = audioSession;
    this.processing = new MicrophoneProcessing(this.reportProcessing, () => audioSession.routing);
    this.reportProcessing(null);
    await this.connect(model, mode, settings, systemPrompt);
  }

  private async connect(model: Model, mode: ConversationMode, settings: object, systemPrompt: string): Promise<void> {
    const signal = this.lifetime.signal;
    const processing = this.processing;
    this.initializing = this.sdk.initDevices();
    await this.initializing;
    if (signal.aborted) return;
    const microphone = this.sdk.tracks().local.audio;
    if (microphone === undefined || microphone.readyState !== "live") throw new Error("No live microphone is available. Check microphone permission and try again.");
    await processing?.configure(microphone);
    if (signal.aborted) return;
    this.sdk.enableMic(true);
    const response = await fetch("/api/conversations", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ model: model.id, mode, settings, system_prompt: systemPrompt }), signal });
    if (!response.ok) {
      const failure: unknown = await response.json().catch(() => null);
      throw new Error(errorMessage(failure, response.status));
    }
    const raw: unknown = await response.json();
    const created = decodeCreated(raw);
    if (signal.aborted) {
      await fetch(`/api/conversations/${created.sessionId}`, { method: "DELETE", keepalive: true });
      return;
    }
    this.sessionId = created.sessionId;
    await Promise.race([
      this.sdk.connect({ webrtcRequestParams: { endpoint: created.offerUrl } }),
      new Promise<void>(resolve => signal.addEventListener("abort", () => resolve(), { once: true })),
    ]);
    if (!signal.aborted) processing?.verify();
  }

  mute(muted: boolean): void { this.sdk.enableMic(!muted); }
  async interrupt(): Promise<void> {
    if (this.sessionId !== null) await fetch(`/api/conversations/${this.sessionId}/interrupt`, { method: "POST", signal: this.lifetime.signal });
  }
  stop(): Promise<void> {
    this.closing ??= this.close();
    return this.closing;
  }
  private async release(): Promise<void> {
    const id = this.sessionId;
    this.sessionId = null;
    if (id !== null) await fetch(`/api/conversations/${id}`, { method: "DELETE", keepalive: true });
  }
  private async close(): Promise<void> {
    this.lifetime.abort();
    this.processing?.close(); this.processing = null;
    this.audio.pause(); this.audio.srcObject = null;
    try {
      // Cancel signaling first, then wait for any late microphone permission.
      try { await this.sdk.disconnect(); }
      finally {
        await this.initializing?.catch(() => {});
        for (const track of Object.values(this.sdk.tracks().local)) track?.stop();
      }
      await this.sdk.disconnect();
    } finally {
      this.audioSession?.close(); this.audioSession = null;
      this.active = false;
      try { await this.release(); } finally { this.ended(); }
    }
  }
}
