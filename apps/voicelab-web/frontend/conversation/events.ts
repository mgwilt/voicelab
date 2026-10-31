/** Runtime validation for application messages crossing the RTVI boundary. */
export type { ConversationMode } from "../domain/session.js";
export interface Notice {
  readonly type: "state" | "text" | "metrics" | "error" | "interrupted";
  readonly text: string;
  readonly speaker: "user" | "assistant";
  readonly final: boolean;
  readonly textMode: "delta" | "snapshot";
  readonly turn: number;
  readonly stage: string;
  readonly seconds: number | null;
  readonly outputRtf: number | null;
  readonly inputRtf: number | null;
}
function number(value: unknown): number | null {
  if (value === null) return null;
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) throw new Error("Invalid timing value");
  return value;
}
export function decodeNotice(value: unknown): Notice {
  if (typeof value !== "object" || value === null || !("type" in value) || !("text" in value) || !("speaker" in value) || !("final" in value) || !("turn" in value) || !("stage" in value) || !("seconds" in value) || !("output_rtf" in value) || !("input_rtf" in value)) throw new Error("Invalid conversation event");
  const { type, text, speaker, final, turn, stage } = value;
  if ((type !== "state" && type !== "text" && type !== "metrics" && type !== "error" && type !== "interrupted") || typeof text !== "string" || (speaker !== "user" && speaker !== "assistant") || typeof final !== "boolean" || typeof turn !== "number" || !Number.isInteger(turn) || turn < 0 || typeof stage !== "string") throw new Error("Invalid conversation event");
  const textMode = "text_mode" in value ? value.text_mode : "delta";
  if (textMode !== "delta" && textMode !== "snapshot") throw new Error("Invalid text update mode");
  return { type, text, speaker, final, textMode, turn, stage, seconds: number(value.seconds), outputRtf: number(value.output_rtf), inputRtf: number(value.input_rtf) };
}
export function decodeCreated(value: unknown): { sessionId: string; offerUrl: string } {
  if (typeof value !== "object" || value === null || !("session_id" in value) || !("offer_url" in value) || typeof value.session_id !== "string" || typeof value.offer_url !== "string" || !/^[A-Za-z0-9_-]+$/.test(value.session_id) || value.offer_url !== `/api/conversations/${value.session_id}/offer`) throw new Error("Invalid conversation reservation");
  return { sessionId: value.session_id, offerUrl: value.offer_url };
}

export function errorMessage(value: unknown, status: number): string {
  if (typeof value === "object" && value !== null && "detail" in value) {
    const detail: unknown = value.detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      const messages: string[] = [];
      for (const item of detail) {
        if (typeof item === "object" && item !== null && "msg" in item && typeof item.msg === "string") messages.push(item.msg);
      }
      if (messages.length > 0) return messages.slice(0, 3).join(". ");
    }
  }
  return `Could not start the conversation (${status}). Try again.`;
}
