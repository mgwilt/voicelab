import { PipecatClient } from "@pipecat-ai/client-js";
import { SmallWebRTCTransport } from "@pipecat-ai/small-webrtc-transport";
import { decodeCreated, decodeNotice, errorMessage, type ConversationMode, type Notice } from "./events.js";
import type { Model } from "../domain/session.js";

export interface ConversationConnection {
  readonly audio: HTMLAudioElement;
  start(model: Model, mode: ConversationMode, settings: object, systemPrompt: string): Promise<void>;
  mute(muted: boolean): void;
  interrupt(): Promise<void>;
  stop(): Promise<void>;
}
export type ConnectionFactory = (event: (event: Notice) => void, status: (message: string) => void, ended: () => void) => ConversationConnection;

/** Owns one microphone, remote audio element, and server reservation. */
export class ConversationClient {
  private readonly sdk: PipecatClient;
  private readonly lifetime = new AbortController();
  private sessionId: string | null = null;
  private closing: Promise<void> | null = null;
  readonly audio = document.createElement("audio");

  constructor(private readonly event: (event: Notice) => void, private readonly status: (message: string) => void, private readonly ended: () => void) {
    this.audio.autoplay = true;
    this.audio.controls = true;
    this.audio.setAttribute("aria-label", "Conversation audio");
    this.sdk = new PipecatClient({
      transport: new SmallWebRTCTransport({ iceServers: [], waitForICEGathering: true }),
      enableMic: true, enableCam: false,
      callbacks: {
        onServerMessage: (value: unknown) => {
          if (this.lifetime.signal.aborted) return;
          try { this.event(decodeNotice(value)); } catch (error) { this.status(String(error)); void this.stop().catch(error => this.status(String(error))); }
        },
        onTrackStarted: (track, participant) => {
          if (this.lifetime.signal.aborted) { track.stop(); return; }
          if (track.kind === "audio" && participant?.local !== true) {
            this.audio.srcObject = new MediaStream([track]);
            void this.audio.play().catch(() => this.status("Tap the audio play button to hear the conversation."));
          }
        },
        onDisconnected: () => { if (!this.lifetime.signal.aborted) { this.status("Conversation ended"); void this.stop().catch(error => this.status(String(error))); } },
        onError: () => { if (this.lifetime.signal.aborted) return; this.status("The voice connection failed. End the call and try again."); void this.stop().catch(error => this.status(String(error))); },
      },
    });
  }

  async start(model: Model, mode: ConversationMode, settings: object, systemPrompt: string): Promise<void> {
    const response = await fetch("/api/conversations", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ model: model.id, mode, settings, system_prompt: systemPrompt }), signal: this.lifetime.signal });
    if (!response.ok) {
      const failure: unknown = await response.json().catch(() => null);
      throw new Error(errorMessage(failure, response.status));
    }
    const raw: unknown = await response.json();
    const created = decodeCreated(raw);
    this.sessionId = created.sessionId;
    if (this.lifetime.signal.aborted) { await this.release(); return; }
    await this.sdk.connect({ webrtcRequestParams: { endpoint: created.offerUrl } });
    if (this.lifetime.signal.aborted) await this.sdk.disconnect();
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
    this.audio.pause(); this.audio.srcObject = null;
    try { await this.sdk.disconnect(); } finally {
      try { await this.release(); } finally { this.ended(); }
    }
  }
}
