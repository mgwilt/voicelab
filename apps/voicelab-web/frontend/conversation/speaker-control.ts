import { element } from "../components/dom.js";

/** Controls local playback only; never changes capture, signaling, or inference. */
export class SpeakerControl {
  readonly element = element("button", "button-secondary", "Mute speaker");
  private readonly lifetime = new AbortController();
  private binding: AbortController | null = null;
  private audio: HTMLAudioElement | null = null;
  private muted = false;

  constructor() {
    this.element.type = "button";
    this.element.addEventListener("click", () => {
      if (this.audio === null) return;
      this.muted = !this.audio.muted;
      this.audio.muted = this.muted;
      this.render();
    }, { signal: this.lifetime.signal });
    this.render();
  }

  bind(audio: HTMLAudioElement | null): void {
    this.binding?.abort(); this.binding = null;
    this.audio = audio;
    if (audio !== null) {
      // Retain an explicit speaker mute across calls in this view.
      audio.muted = this.muted;
      this.binding = new AbortController();
      audio.addEventListener("volumechange", () => {
        this.muted = audio.muted;
        this.render();
      }, { signal: this.binding.signal });
    }
    this.render();
  }

  private render(): void {
    this.element.disabled = this.audio === null;
    this.element.textContent = this.muted ? "Unmute speaker" : "Mute speaker";
    this.element.setAttribute("aria-pressed", String(this.muted));
  }

  dispose(): void { this.bind(null); this.lifetime.abort(); this.element.remove(); }
}
