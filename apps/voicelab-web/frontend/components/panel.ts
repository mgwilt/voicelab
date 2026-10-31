import type { Emit, VoiceSession } from "../domain/session.js";

export interface ControlPanel {
  readonly element: HTMLElement;
  createSession(emit: Emit): VoiceSession;
  setDisabled(disabled: boolean): void;
  dispose(): void;
}
