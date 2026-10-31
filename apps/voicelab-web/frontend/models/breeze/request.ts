export interface SynthesisRequest {
  readonly text: string;
  readonly instruction: string;
  readonly seed: number;
  readonly cfg_scale: number;
  readonly max_frames: number;
}
