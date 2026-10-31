import type { ReferenceAudio } from "../synthesis/reference.js";

export interface Request {
  readonly text: string;
  readonly voice: string | null;
  readonly preset: "stable" | "expressive";
  readonly max_tokens: number;
  readonly ref_audio: ReferenceAudio | null;
  readonly ref_text: string | null;
  readonly use_reference_prompt: boolean | null;
  readonly temperature: number | null;
  readonly top_k: number | null;
  readonly top_p: number | null;
  readonly min_p: number | null;
  readonly repetition_penalty: number;
  readonly repetition_window: number | null;
  readonly token_run_penalty: number;
  readonly token_run_grace: number;
  readonly chunk_chars: number | null;
  readonly chunk_min_chars: number | null;
  readonly use_emotion: boolean | null;
  readonly seed: number;
  readonly normalize: boolean;
}
