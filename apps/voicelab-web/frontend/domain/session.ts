export type AddedSynthesisId = "phonon" | "kokoro" | "audio8" | "ming_tts" | "drama" | "maya" | "omnivoice" | "kitten";
export type ModelId = "nemotron" | "sesame" | "breeze" | AddedSynthesisId | "personaplex" | "ming_audio" | "qwen_omni" | "minicpm";

export type ConversationMode = "cascade" | "turn" | "duplex";

interface ModelBase {
  readonly conversationModes?: readonly ConversationMode[];
  readonly label: string;
  readonly description: string;
  readonly backend: string;
  readonly available: boolean;
  readonly reason: string | null;
}
export interface DuplexModel extends ModelBase {
  readonly id: "nemotron";
  readonly kind: "duplex";
  readonly endpoint: "/api/nemotron/session";
  readonly inputSampleRate: 16000;
  readonly outputSampleRate: 22050;
}
export interface SesameModel extends ModelBase {
  readonly id: "sesame";
  readonly kind: "synthesis";
  readonly endpoint: "/api/sesame/session";
  readonly outputSampleRate: 24000;
}
export interface BreezeModel extends ModelBase {
  readonly id: "breeze";
  readonly kind: "synthesis";
  readonly endpoint: "/api/breeze/session";
  readonly outputSampleRate: 24000;
}
interface SynthesisRates {
  phonon: 24000; kokoro: 24000; audio8: 44100; ming_tts: 44100;
  drama: 44100; maya: 24000; omnivoice: 24000; kitten: 24000;
}
export type AddedSynthesisModel = {
  [Id in AddedSynthesisId]: ModelBase & { readonly id: Id; readonly kind: "synthesis";
    readonly endpoint: `/api/${Id}/session`; readonly outputSampleRate: SynthesisRates[Id] }
}[AddedSynthesisId];
export interface PersonaPlexModel extends ModelBase {
  readonly id: "personaplex"; readonly kind: "duplex";
  readonly endpoint: "/api/personaplex/session";
  readonly inputSampleRate: 24000; readonly outputSampleRate: 24000;
}
export interface MingAudioModel extends ModelBase {
  readonly id: "ming_audio"; readonly kind: "conversation";
  readonly endpoint: "/api/ming_audio/session"; readonly outputSampleRate: 44100;
}
export interface QwenOmniModel extends ModelBase {
  readonly id: "qwen_omni"; readonly kind: "conversation";
  readonly endpoint: "/api/qwen_omni/session"; readonly outputSampleRate: 24000;
}
export interface MiniCPMModel extends ModelBase {
  readonly id: "minicpm"; readonly kind: "duplex";
  readonly endpoint: "/api/minicpm/session";
  readonly inputSampleRate: 16000; readonly outputSampleRate: 24000;
}
export type SynthesisModel = SesameModel | BreezeModel | AddedSynthesisModel;
export type Model = DuplexModel | SynthesisModel | PersonaPlexModel | MingAudioModel | QwenOmniModel | MiniCPMModel;
export type Phase = "idle" | "preparing" | "running" | "draining" | "stopping";
export interface Measurement {
  readonly basis: "input" | "output";
  readonly rtf: number | null;
  readonly audioSeconds: number;
  readonly latencyMs: number | null;
  readonly latencyLabel: "Frame p95" | "First PCM" | "Compute to first PCM";
  readonly backlogSeconds: number | null;
}
export interface Outcome {
  readonly kind: "complete" | "incomplete" | "unverified" | "cancelled";
  readonly message: string;
}
export type SessionEvent =
  | { readonly type: "status"; readonly phase: Phase; readonly message: string }
  | { readonly type: "measurement"; readonly value: Measurement }
  | { readonly type: "transcript"; readonly speaker: "user" | "assistant" | "function"; readonly text: string };
export type Emit = (event: SessionEvent) => void;
export interface VoiceSession {
  run(signal: AbortSignal): Promise<Outcome>;
  stop(): void;
}
export function unreachable(value: never): never {
  throw new Error(`Unhandled variant: ${String(value)}`);
}
