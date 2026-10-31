export type ModelId = "nemotron" | "sesame" | "breeze";

interface ModelBase {
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
export type SynthesisModel = SesameModel | BreezeModel;
export type Model = DuplexModel | SynthesisModel;
export type Phase = "idle" | "preparing" | "running" | "draining" | "stopping";
export interface Measurement {
  readonly basis: "input" | "output";
  readonly rtf: number | null;
  readonly audioSeconds: number;
  readonly latencyMs: number | null;
  readonly latencyLabel: "Frame p95" | "First PCM";
  readonly backlogSeconds: number | null;
}
export interface Outcome {
  readonly kind: "complete" | "incomplete" | "cancelled";
  readonly message: string;
}
export type SessionEvent =
  | { readonly type: "status"; readonly phase: Phase; readonly message: string }
  | { readonly type: "measurement"; readonly value: Measurement }
  | { readonly type: "transcript"; readonly speaker: "user" | "assistant"; readonly text: string };
export type Emit = (event: SessionEvent) => void;
export interface ExperimentSession {
  run(signal: AbortSignal): Promise<Outcome>;
  stop(): void;
}
export function unreachable(value: never): never {
  throw new Error(`Unhandled variant: ${String(value)}`);
}
