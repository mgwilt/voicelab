import type { ReferenceAudio } from "../synthesis/reference.js";

export interface Request {
  readonly text: string;
  readonly language: string | null;
  readonly instruction: string | null;
  readonly steps: number;
  readonly speed: number;
  readonly ref_audio: ReferenceAudio | null;
  readonly ref_text: string | null;
  readonly duration: number | null;
  readonly normalize: boolean;
  readonly denoise: boolean;
  readonly preprocess_prompt: boolean;
  readonly postprocess_output: boolean;
  readonly guidance_scale: number;
  readonly t_shift: number;
  readonly layer_penalty_factor: number;
  readonly position_temperature: number;
  readonly class_temperature: number;
  readonly audio_chunk_duration: number;
  readonly audio_chunk_threshold: number;
  readonly pad_duration: number;
  readonly fade_duration: number;
  readonly seed: number;
}
