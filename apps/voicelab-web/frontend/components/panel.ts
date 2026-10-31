import type { ConversationMode, Emit, VoiceSession } from "../domain/session.js";

export interface ControlPanel {
  readonly element: HTMLElement;
  conversationConfig?(): Promise<object>;
  prepareConversation?(mode?: ConversationMode): void;
  createSession(emit: Emit): VoiceSession;
  setDisabled(disabled: boolean): void;
  dispose(): void;
}
