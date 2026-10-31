import { expect, spyOn, test } from "bun:test";
import { SpeakerControl } from "../frontend/conversation/speaker-control.js";

test("speaker mute changes output without pausing playback, and survives a new call", () => {
  const control = new SpeakerControl();
  const audio = document.createElement("audio");
  const pause = spyOn(audio, "pause");
  expect(control.element.disabled).toBe(true);
  control.bind(audio);
  control.element.click();
  expect(audio.muted).toBe(true);
  expect(pause).not.toHaveBeenCalled();
  expect(control.element.textContent).toBe("Unmute speaker");
  expect(control.element.getAttribute("aria-pressed")).toBe("true");
  control.bind(null);
  expect(control.element.disabled).toBe(true);
  const next = document.createElement("audio");
  control.bind(next);
  expect(next.muted).toBe(true);
  control.element.click();
  expect(next.muted).toBe(false);
  expect(control.element.getAttribute("aria-pressed")).toBe("false");
  control.dispose(); pause.mockRestore();
});

test("native mute changes stay synchronized and detached players cannot alter the preference", () => {
  const control = new SpeakerControl();
  const old = document.createElement("audio");
  control.bind(old);
  old.muted = true;
  old.dispatchEvent(new Event("volumechange"));
  expect(control.element.textContent).toBe("Unmute speaker");
  const current = document.createElement("audio");
  control.bind(current);
  old.muted = false;
  old.dispatchEvent(new Event("volumechange"));
  expect(current.muted).toBe(true);
  expect(control.element.textContent).toBe("Unmute speaker");
  control.dispose();
  current.muted = false;
  current.dispatchEvent(new Event("volumechange"));
  expect(control.element.textContent).toBe("Unmute speaker");
});
