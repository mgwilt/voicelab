import type { ControlPanel } from "../../components/panel.js";
import type { Emit, ExperimentSession, PersonaPlexModel } from "../../domain/experiment.js";
import { SpeechForm } from "../synthesis/form.js";
import { PersonaPlexSession } from "./session.js";

export class PersonaPlexPanel implements ControlPanel {
  private readonly form = new SpeechForm();
  readonly element = this.form.element;
  private readonly prompt = this.form.text("Instructions", "You are a helpful assistant. Answer clearly and concisely.", 2000);
  private readonly voice = this.form.choice("Voice", ["NATF0.pt", "NATF1.pt", "NATF2.pt", "NATF3.pt", "NATM0.pt", "NATM1.pt", "NATM2.pt", "NATM3.pt", "VARF0.pt", "VARF1.pt", "VARF2.pt", "VARF3.pt", "VARF4.pt", "VARM0.pt", "VARM1.pt", "VARM2.pt", "VARM3.pt", "VARM4.pt"], "NATF2.pt");
  private readonly reference = this.form.referenceAudio("Reference voice (replaces preset)");
  private readonly sampling = this.form.section("Sampling controls");
  private readonly seed = this.sampling.integer("Seed", 0, 0, 4294967295);
  private readonly textTemperature = this.sampling.decimal("Text temperature", 0.7, 0, 2);
  private readonly audioTemperature = this.sampling.decimal("Audio temperature", 0.8, 0, 2);
  private readonly textTopK = this.sampling.integer("Text top-k", 25, 1, 32000);
  private readonly audioTopK = this.sampling.integer("Audio top-k", 250, 1, 2048);
  constructor(private readonly model: PersonaPlexModel) {
    this.form.note("Your microphone stays live while the model speaks. Inference runs locally; timing measures model and codec processing.");
  }
  createSession(emit: Emit): ExperimentSession { return new PersonaPlexSession(this.model, async () => ({ type: "start", prompt: this.prompt(), voice: this.voice(), reference_audio: await this.reference(), seed: this.seed(), text_temperature: this.textTemperature(), audio_temperature: this.audioTemperature(), text_top_k: this.textTopK(), audio_top_k: this.audioTopK() }), emit); }
  setDisabled(disabled: boolean): void { this.form.setDisabled(disabled); }
  dispose(): void { this.element.remove(); }
}
