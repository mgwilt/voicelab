export interface Request {
  readonly voice: string;
  readonly temperature: number;
  readonly streaming: boolean;
  readonly text: string;
  readonly seed: number;
  readonly max_frames: number;
}
