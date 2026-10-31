import { AudioInput } from "../../components/audio-input.js";
import type { ControlPanel } from "../../components/panel.js";
import type { Emit, VoiceSession, MingAudioModel } from "../../domain/session.js";
import { SpeechForm } from "../synthesis/form.js";
import { HistoryInput } from "./history.js";
import { MingAudioSession } from "./session.js";

export class MingAudioPanel implements ControlPanel {
  private readonly form = new SpeechForm();
  readonly element = this.form.element;
  private readonly task = this.form.choice("Task", ["spoken_dialogue", "audio_understanding", "asr", "text", "tts"], "spoken_dialogue");
  private readonly audio = new AudioInput("Input audio");
  private readonly prompt = this.form.text("Instructions or text to speak", "Please answer the question in the audio.", 2000);
  private readonly voice = this.form.text("Preset voice", "US_Stella", 80);
  private readonly reference = this.form.referenceAudio("Reference voice (replaces preset)");
  private readonly trimReference = this.form.checkbox("Trim trailing silence for reference continuation", true);
  private readonly referenceText = this.form.optionalText("Reference transcript", 2000);
  private readonly advanced = this.form.section("Reasoning and conversation context");
  private readonly tokens = this.advanced.integer("Token limit", 256, 1, 2048);
  private readonly thinking = this.advanced.checkbox("Enable reasoning", false);
  private readonly system = this.advanced.optionalText("System instructions", 2000);
  private readonly history = new HistoryInput();
  private readonly sampling = this.form.section("Speech generation");
  private readonly seed = this.sampling.integer("Seed", 0, 0, 4294967295);
  private readonly decodeSteps = this.sampling.integer("Speech step limit", 375, 1, 1500);
  private readonly cfg = this.sampling.decimal("Guidance", 2, 0, 10);
  private readonly sigma = this.sampling.decimal("Noise scale", 0.25, 0, 1);
  private readonly temperature = this.sampling.decimal("Temperature", 0, 0, 2);
  private readonly flowSteps = this.sampling.integer("Flow steps", 10, 1, 100);
  constructor(private readonly model: MingAudioModel) {
    this.element.prepend(this.audio.element);
    this.advanced.element.append(this.history.element);
    this.form.note("Spoken dialogue and TTS produce buffered speech. ASR transcribes audio; audio understanding and text return text only. A reference voice needs at least 0.4 seconds.");
  }
  createSession(emit: Emit): VoiceSession {
    const file = this.audio.read();
    const task = this.task();
    if ((task === "asr" || task === "audio_understanding") && file === null) throw new Error("This task needs input audio.");
    if (task === "tts" && file !== null) throw new Error("Remove input audio for TTS; use the reference voice field for voice conditioning.");
    return new MingAudioSession(this.model, { file, task, prompt: this.prompt(), voice: this.voice(), max_tokens: this.tokens(),
      seed: this.seed(), max_decode_steps: this.decodeSteps(), cfg_scale: this.cfg(), sigma: this.sigma(), temperature: this.temperature(), flow_steps: this.flowSteps(),
      reference: this.reference, reference_text: this.referenceText(), trim_reference_silence: this.trimReference(), thinking: this.thinking(), system_prompt: this.system(), history: () => this.history.read() }, emit);
  }
  setDisabled(disabled: boolean): void { this.form.setDisabled(disabled); this.audio.setDisabled(disabled); this.history.setDisabled(disabled); }
  dispose(): void { this.audio.dispose(); this.element.remove(); }
}
