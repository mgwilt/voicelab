import type { ReferenceAudio } from "../synthesis/reference.js";
export interface Request {
  readonly text: string;
  readonly instruction: string | null;
  readonly language: "zh" | "en" | "ja" | "ko" | "de" | "fr" | "es";
  readonly ref_audio: ReferenceAudio | null;
  readonly ref_text: string | null;
  readonly trim_reference_silence: boolean;
  readonly voice: string | null;
  readonly style: string | null;
  readonly emotion: "高兴" | "悲伤" | "愤怒" | "恐惧" | "厌恶" | "惊讶" | "中性" | null;
  readonly dialect: "cantonese" | "chuanyu dialect" | null;
  readonly speed: "慢速" | "中速" | "快速" | null;
  readonly pitch: "低" | "中" | "高" | null;
  readonly volume: "低" | "中" | "高" | null;
  readonly seed: number;
  readonly max_frames: number;
  readonly cfg_scale: number;
  readonly flow_steps: number;
  readonly temperature: number;
  readonly sigma: number;
}
