import type { ReferenceAudio } from "../synthesis/reference.js";

export interface SynthesisRequest {
  readonly text: string;
  readonly instruction: string | null;
  readonly seed: number;
  readonly cfg_scale: number;
  readonly max_frames: number;
  readonly ref_audio: ReferenceAudio | null;
  readonly ref_text: string | null;
  readonly temperature: number;
  readonly top_k: number;
  readonly top_p: number;
}
