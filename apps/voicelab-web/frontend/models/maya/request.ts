import type { ReferenceAudio } from "../synthesis/reference.js";

export interface Request {
  readonly text: string;
  readonly language: "as" | "bn" | "en" | "gu" | "hi" | "kn" | "ml" | "mr" | "or" | "pa" | "ta" | "te";
  readonly voice: "Ahana" | "Vaidehi" | "Malini" | "Charvi" | "Jahnavi" | "Anirban" | "Nakul" | "Vasant" | "Siddhant" | "Satvik" | "Abhay" | "Nivedita" | "Ishwari" | "Ritvik" | "Vrinda" | "Sreekar" | "Kayal" | "Shaunak" | "Akshat" | "Tanushree" | null;
  readonly ref_audio: ReferenceAudio | null;
  readonly ref_text: string | null;
  readonly ref_language: "as" | "bn" | "en" | "gu" | "hi" | "kn" | "ml" | "mr" | "or" | "pa" | "ta" | "te" | null;
  readonly conditioning: "icl" | "spk" | "icl_spk";
  readonly temperature: number;
  readonly top_k: number;
  readonly top_p: number;
  readonly depth_temperature: number;
  readonly depth_top_k: number;
  readonly presence_penalty: number;
  readonly presence_window: number;
  readonly min_frames: number;
  readonly streaming_interval: number;
  readonly max_frames: number;
  readonly seed: number;
  readonly normalize: boolean;
}
