import type { ReferenceAudio } from "../synthesis/reference.js";
import type { HistoryTurn } from "./history.js";
export interface Request {
  readonly audio: ReferenceAudio | null;
  readonly prompt: string; readonly system_prompt: string;
  readonly history: readonly HistoryTurn[];
  readonly speaker: "Ethan" | "Chelsie" | "Aiden";
  readonly return_audio: boolean; readonly stream_audio: boolean;
  readonly seed: number; readonly max_tokens: number; readonly temperature: number;
  readonly top_p: number; readonly top_k: number;
  readonly talker_max_tokens: number; readonly talker_temperature: number;
  readonly talker_top_p: number; readonly talker_top_k: number;
  readonly talker_repetition_penalty: number; readonly chunk_frames: number;
}
export async function inputAudio(file: File | null): Promise<ReferenceAudio | null> {
  if (file === null) return null;
  const bytes = new Uint8Array(await file.arrayBuffer());
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return { wav_base64: btoa(binary) };
}
