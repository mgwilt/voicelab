import { element, integerInput, labeled, readInteger } from "../../components/dom.js";
import type { ControlPanel } from "../../components/panel.js";
import type { Emit, ExperimentSession, SynthesisModel } from "../../domain/experiment.js";
import { SynthesisSession } from "./session.js";
import { ReferenceAudioInput, type ReferenceAudio } from "./reference.js";

/** View controls only. Each model owns its request construction and constraints. */
export class SpeechForm {
  readonly element = element("div", "control-panel");
  private readonly inputs: (HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement)[] = [];
  private readonly references: ReferenceAudioInput[] = [];
  private readonly sections: SpeechForm[] = [];
  section(label: string): SpeechForm {
    const section = new SpeechForm();
    const details = element("details");
    details.append(element("summary", "", label), section.element);
    this.sections.push(section); this.element.append(details);
    return section;
  }
  optionalText(label: string, maxLength: number): () => string | null {
    const input = element("textarea");
    input.maxLength = maxLength; input.rows = 2;
    this.inputs.push(input); this.element.append(labeled(label, input));
    return () => {
      if (input.value.length > maxLength) throw new Error(`${label} exceeds ${maxLength} characters.`);
      return input.value.trim() === "" ? null : input.value;
    };
  }
  optionalDecimal(label: string, min: number, max: number): () => number | null {
    const input = element("input");
    input.type = "number"; input.min = String(min); input.max = String(max); input.step = "any";
    this.inputs.push(input); this.element.append(labeled(label, input, "Blank uses the model default."));
    return () => {
      if (input.value === "") return null;
      if (!input.checkValidity() || !Number.isFinite(input.valueAsNumber)) throw new Error(`Invalid ${label.toLowerCase()}.`);
      return input.valueAsNumber;
    };
  }
  optionalInteger(label: string, min: number, max: number): () => number | null {
    const read = this.optionalDecimal(label, min, max);
    return () => {
      const value = read();
      if (value !== null && !Number.isSafeInteger(value)) throw new Error(`${label} must be a whole number.`);
      return value;
    };
  }
  referenceAudio(label: string): () => Promise<ReferenceAudio | null> {
    const input = new ReferenceAudioInput(label);
    this.references.push(input); this.element.append(input.element);
    return () => input.read();
  }
  text(label: string, value: string, maxLength: number): () => string {
    const input = element("textarea");
    input.value = value; input.maxLength = maxLength; input.rows = 3; input.required = true;
    this.inputs.push(input); this.element.append(labeled(label, input));
    return () => {
      if (input.value.trim() === "" || input.value.length > maxLength) throw new Error(`Enter ${label.toLowerCase()} within ${maxLength} characters.`);
      return input.value;
    };
  }
  integer(label: string, value: number, min: number, max: number): () => number {
    const input = integerInput(value, min, max);
    this.inputs.push(input); this.element.append(labeled(label, input));
    return () => readInteger(input);
  }
  decimal(label: string, value: number, min: number, max: number): () => number {
    const input = element("input");
    input.type = "number"; input.value = String(value); input.min = String(min); input.max = String(max); input.step = "any"; input.required = true;
    this.inputs.push(input); this.element.append(labeled(label, input));
    return () => {
      if (!input.checkValidity() || !Number.isFinite(input.valueAsNumber)) throw new Error(`Invalid ${label.toLowerCase()}.`);
      return input.valueAsNumber;
    };
  }
  choice<Value extends string>(label: string, values: readonly Value[], initial: Value): () => Value {
    const input = element("select");
    for (const value of values) { const option = element("option", "", value); option.value = value; input.append(option); }
    input.value = initial;
    this.inputs.push(input); this.element.append(labeled(label, input));
    return () => {
      const value = values.find(value => value === input.value);
      if (value === undefined) throw new Error(`Invalid ${label.toLowerCase()}.`);
      return value;
    };
  }
  checkbox(label: string, initial: boolean): () => boolean {
    const input = element("input");
    input.type = "checkbox"; input.checked = initial;
    this.inputs.push(input); this.element.append(labeled(label, input));
    return () => input.checked;
  }
  note(message: string): void { this.element.append(element("p", "workflow-note", message)); }
  setDisabled(disabled: boolean): void {
    for (const input of this.inputs) input.disabled = disabled;
    for (const input of this.references) input.setDisabled(disabled);
    for (const section of this.sections) section.setDisabled(disabled);
  }
}

export class SpeechPanel<Request extends object> implements ControlPanel {
  private readonly form = new SpeechForm();
  readonly element = this.form.element;
  private readonly request: () => Request | Promise<Request>;
  constructor(private readonly model: SynthesisModel, build: (form: SpeechForm) => () => Request | Promise<Request>) {
    this.request = build(this.form);
  }
  createSession(emit: Emit): ExperimentSession { return new SynthesisSession({ model: this.model, request: this.request }, emit); }
  setDisabled(disabled: boolean): void { this.form.setDisabled(disabled); }
  dispose(): void { this.element.remove(); }
}
