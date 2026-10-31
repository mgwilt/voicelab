import type { ReferenceAudio } from "../synthesis/reference.js";

export interface Reference { readonly audio: ReferenceAudio; readonly text: string; }

export interface Request {
  readonly text: string;
  readonly references: readonly Reference[];
  readonly previous_takes: readonly Reference[];
  readonly top_k: number | null;
  readonly streaming_interval: number;
  readonly first_chunk_frames: number | null;
  readonly seed: number;
  readonly max_new_tokens: number;
  readonly temperature: number;
  readonly top_p: number;
}
