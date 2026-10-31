import { hideFields } from "../../conversation/settings.js";
import { AudioInput } from "../../components/audio-input.js";
import type { ControlPanel } from "../../components/panel.js";
import type { ConversationMode, Emit, VoiceSession, MiniCPMModel } from "../../domain/session.js";
import { SpeechForm } from "../synthesis/form.js";
import type { ReferenceAudio } from "../synthesis/reference.js";
import { MiniCPMSession, type StartRequest } from "./session.js";
async function audioPayload(file: File | null): Promise<ReferenceAudio | null> {
  if (file === null) return null;
  const bytes = new Uint8Array(await file.arrayBuffer());
  let binary = ""; for (const byte of bytes) binary += String.fromCharCode(byte);
  return { wav_base64: btoa(binary) };
}
export class MiniCPMPanel implements ControlPanel {
  private readonly form = new SpeechForm();
  readonly element = this.form.element;
  private readonly mode = this.form.choice("Mode", ["Live conversation", "Spoken answer"], "Live conversation");
  private readonly prompt = this.form.text("Instructions", "You are a helpful voice assistant. Answer in English, briefly and naturally.", 2000);
  private readonly turn = this.form.section("Spoken answer input");
  private readonly text = this.turn.optionalText("Message (optional with audio)", 1200);
  private readonly audio = new AudioInput("Recorded message");
  private readonly reference = this.form.referenceAudio("Audio context reference (optional, 1–30 seconds)");
  private readonly historyForm = this.form.section("Previous conversation (spoken answers only)");
  private readonly history = Array.from({ length: 3 }, (_, index) => ({
    user: this.historyForm.optionalText(`Previous message ${index + 1}`, 1200),
    assistant: this.historyForm.optionalText(`Previous answer ${index + 1}`, 1200),
  }));
  private readonly sampling = this.form.section("Generation controls");
  private readonly seed = this.sampling.integer("Seed", 0, 0, 4294967295);
  private readonly temperature = this.sampling.decimal("Text temperature", 0.7, 0, 2);
  private readonly topP = this.sampling.decimal("Text top-p", 0.8, 0.0001, 1);
  private readonly topK = this.sampling.integer("Text top-k", 100, 1, 200);
  private readonly ttsTemperature = this.sampling.decimal("Speech temperature", 0.8, 0.0001, 2);
  private readonly lengthPenalty = this.sampling.decimal("Speech length penalty", 1.1, 0.1, 2);
  private readonly liveControls = this.sampling.section("Live controls");
  private readonly listenScale = this.liveControls.decimal("Listen probability scale (live)", 1, 0.0001, 3);
  private readonly speakTokens = this.liveControls.integer("Speech tokens per chunk (live)", 20, 1, 100);
  private readonly turnControls = this.sampling.section("Spoken answer controls");
  private readonly maxTokens = this.turnControls.integer("Response token limit (spoken answer)", 256, 1, 2048);
  private readonly listeners = new AbortController();
  private disabled = false;
  private conversationMode: ConversationMode | null = null;
  constructor(private readonly model: MiniCPMModel) {
    this.turn.element.append(this.audio.element);
    this.element.addEventListener("change", () => this.updateMode(), { signal: this.listeners.signal });
    this.updateMode();
    this.form.note("Live mode keeps listening during speech, for up to two minutes. Spoken answer uses recorded audio or text. The first run builds the native runtime and downloads about 8 GB. Reference audio conditions understanding; the native waveform decoder keeps its preset speaker. Voice only; image and video input are outside this lab workflow.");
  }
  prepareConversation(mode: ConversationMode = "duplex"): void {
    this.conversationMode = mode;
    hideFields(this.element, ["Mode"]);
    this.updateMode();
  }
  async conversationConfig(): Promise<object> { return { mode: "full_duplex", prompt: this.prompt(), reference_audio: await this.reference(), seed: this.seed(), temperature: this.temperature(), top_p: this.topP(), top_k: this.topK(), tts_temperature: this.ttsTemperature(), length_penalty: this.lengthPenalty(), listen_prob_scale: this.listenScale(), max_speak_tokens: this.speakTokens(), max_tokens: this.maxTokens() }; }
  createSession(emit: Emit): VoiceSession {
    return new MiniCPMSession(this.model, async (): Promise<StartRequest> => {
      const live = this.mode() === "Live conversation";
      const history: { role: "user" | "assistant"; text: string }[] = [];
      let gap = false;
      if (!live) for (const pair of this.history) {
        const user = pair.user(); const assistant = pair.assistant();
        if (user === null && assistant === null) { gap = true; continue; }
        if (gap || user === null || assistant === null) throw new Error("Fill previous messages and answers in order, with both parts of each turn.");
        history.push({ role: "user", text: user }, { role: "assistant", text: assistant });
      }
      const text = live ? "" : this.text() ?? "";
      const audio = live ? null : await audioPayload(this.audio.read());
      if (!live && text === "" && audio === null) throw new Error("Record a message, upload audio, or enter text.");
      return { type: "start", mode: live ? "full_duplex" : "turn_based", prompt: this.prompt(), text, audio, history,
        reference_audio: await this.reference(), seed: this.seed(), temperature: this.temperature(),
        top_p: this.topP(), top_k: this.topK(), tts_temperature: this.ttsTemperature(),
        length_penalty: this.lengthPenalty(), listen_prob_scale: this.listenScale(),
        max_speak_tokens: this.speakTokens(), max_tokens: this.maxTokens() };
    }, emit);
  }
  private updateMode(): void {
    const live = this.conversationMode === null ? this.mode() === "Live conversation" : this.conversationMode === "duplex";
    for (const section of [this.turn, this.historyForm]) {
      if (section.element.parentElement !== null) section.element.parentElement.hidden = live || this.conversationMode !== null;
    }
    if (this.turnControls.element.parentElement !== null) this.turnControls.element.parentElement.hidden = live;
    if (this.liveControls.element.parentElement !== null) this.liveControls.element.parentElement.hidden = !live;
    this.audio.setDisabled(this.disabled || live || this.conversationMode !== null);
  }
  setDisabled(disabled: boolean): void { this.disabled = disabled; this.form.setDisabled(disabled); this.updateMode(); }
  dispose(): void { this.listeners.abort(); this.audio.dispose(); this.element.remove(); }
}
