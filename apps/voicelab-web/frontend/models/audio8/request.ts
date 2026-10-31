import type { ReferenceAudio } from "../synthesis/reference.js";

export interface Request {
  readonly ref_audio: ReferenceAudio | null;
  readonly ref_text: string | null;
  readonly top_p: number;
  readonly top_k: number;
  readonly do_sample: boolean;
  readonly text: string;
  readonly seed: number;
  readonly temperature: number;
  readonly max_frames: number;
}
