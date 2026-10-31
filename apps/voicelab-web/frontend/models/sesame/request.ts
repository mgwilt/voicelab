import type { ReferenceAudio } from "../synthesis/reference.js";

export interface ContextSegment {
  readonly audio: ReferenceAudio;
  readonly text: string;
  readonly speaker: 0 | 1;
}

export interface SynthesisRequest {
  readonly text: string;
  readonly speaker: 0 | 1;
  readonly seed: number;
  readonly max_frames: number;
  readonly ref_audio: ReferenceAudio | null;
  readonly ref_text: string | null;
  readonly temperature: number;
  readonly top_k: number;
  readonly context: readonly ContextSegment[];
}
