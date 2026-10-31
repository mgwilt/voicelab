import { AudioInput } from "../../components/audio-input.js";
import type { ControlPanel } from "../../components/panel.js";
import type { Emit, ExperimentSession, QwenOmniModel } from "../../domain/experiment.js";
import { SpeechForm } from "../synthesis/form.js";
import { HistoryInput } from "./history.js";
import { inputAudio } from "./request.js";
import { QwenOmniSession } from "./session.js";
export class QwenOmniPanel implements ControlPanel {
  private readonly form = new SpeechForm();
  readonly element = this.form.element;
  private readonly audio = new AudioInput("Question audio (optional)");
  private readonly prompt = this.form.text("Instructions or question", "Answer briefly.", 4000);
  private readonly speaker = this.form.choice("Voice", ["Ethan", "Chelsie", "Aiden"], "Ethan");
  private readonly speech = this.form.checkbox("Speak the answer", true);
  private readonly streaming = this.form.checkbox("Stream generated speech", true);
  private readonly context = this.form.section("System instructions and previous exchanges");
  private readonly system = this.context.text("System instructions", "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.", 4000);
  private readonly history = new HistoryInput();
  private readonly generation = this.form.section("Generation controls");
  private readonly seed = this.generation.integer("Seed", 0, 0, 4294967295);
  private readonly tokens = this.generation.integer("Text token limit", 256, 1, 2048);
  private readonly temperature = this.generation.decimal("Text temperature", 0, 0, 2);
  private readonly topP = this.generation.decimal("Text top P", 1, 0.001, 1);
  private readonly topK = this.generation.integer("Text top K", 0, 0, 200);
  private readonly speechTokens = this.generation.integer("Speech frame limit", 1500, 1, 1500);
  private readonly speechTemperature = this.generation.decimal("Speech temperature", 0.9, 0, 2);
  private readonly speechP = this.generation.decimal("Speech top P", 1, 0.001, 1);
  private readonly speechK = this.generation.integer("Speech top K", 50, 0, 200);
  private readonly repetition = this.generation.decimal("Speech repetition penalty", 1.05, 1, 2);
  private readonly chunk = this.generation.integer("Frames per audio chunk", 25, 1, 300);
  constructor(private readonly model: QwenOmniModel) {
    this.element.prepend(this.audio.element);
    this.context.element.append(this.history.element);
    this.form.note("Record or upload a question, or use text. Qwen finishes its text response before generating speech. This is turn-based conversation.");
  }
  createSession(emit: Emit): ExperimentSession {
    const file = this.audio.read();
    const values = { prompt: this.prompt(), system_prompt: this.system(), speaker: this.speaker(), return_audio: this.speech(), stream_audio: this.streaming(), seed: this.seed(), max_tokens: this.tokens(), temperature: this.temperature(), top_p: this.topP(), top_k: this.topK(), talker_max_tokens: this.speechTokens(), talker_temperature: this.speechTemperature(), talker_top_p: this.speechP(), talker_top_k: this.speechK(), talker_repetition_penalty: this.repetition(), chunk_frames: this.chunk() };
    return new QwenOmniSession(this.model, async () => ({ ...values, audio: await inputAudio(file), history: await this.history.read() }), emit);
  }
  setDisabled(disabled: boolean): void { this.audio.setDisabled(disabled); this.form.setDisabled(disabled); this.history.setDisabled(disabled); }
  dispose(): void { this.audio.dispose(); this.element.remove(); }
}
