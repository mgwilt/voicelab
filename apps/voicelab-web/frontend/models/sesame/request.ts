export interface SynthesisRequest {
  readonly text: string;
  readonly speaker: 0 | 1;
  readonly seed: number;
  readonly max_frames: number;
}
