import { element, labeled } from "../../components/dom.js";

export interface ReferenceAudio { readonly wav_base64: string; }

/** A file is read only when a run starts; no local filesystem paths cross the wire. */
export class ReferenceAudioInput {
  private readonly input = element("input");
  readonly element: HTMLLabelElement;
  constructor(label: string) {
    this.input.type = "file";
    this.input.accept = ".wav,audio/wav";
    this.element = labeled(label, this.input, "Optional mono PCM16 WAV, 8–48 kHz, at most 30 seconds.");
  }
  async read(): Promise<ReferenceAudio | null> {
    const file = this.input.files?.item(0);
    if (file === undefined || file === null) return null;
    if (file.size > 3 * 1024 * 1024) throw new Error("Reference WAV exceeds 3 MiB.");
    const bytes = new Uint8Array(await file.arrayBuffer());
    let binary = "";
    for (const byte of bytes) binary += String.fromCharCode(byte);
    return { wav_base64: btoa(binary) };
  }
  setDisabled(disabled: boolean): void { this.input.disabled = disabled; }
  dispose(): void { this.element.remove(); }
}
