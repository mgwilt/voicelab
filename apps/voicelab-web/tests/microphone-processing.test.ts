import { expect, test } from "bun:test";
import { MicrophoneProcessing, type MicrophoneProcessingReport } from "../frontend/audio/processing.js";
import { MicrophoneStatus } from "../frontend/conversation/microphone-status.js";
import { microphone } from "./microphone-fixture.js";

test("requests processing on the capture track, preserves device constraints, and reports actual settings", async () => {
  const reports: MicrophoneProcessingReport[] = [];
  let applied: MediaTrackConstraints | undefined;
  const track = microphone({ echoCancellation: false, noiseSuppression: true }, async value => { applied = value; }, { deviceId: { exact: "selected" }, channelCount: 1 });
  const processing = new MicrophoneProcessing(report => { if (report !== null) reports.push(report); });
  await processing.configure(track);
  expect(applied).toEqual({ deviceId: { exact: "selected" }, channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true });
  expect(reports).toEqual([{ callAudioRouting: "unavailable", echoCancellation: false, noiseSuppression: true, autoGainControl: null, configurationFailed: false }]);
  processing.close();
});

test("unsupported or unreported processing remains unknown, and constraint rejection is visible", async () => {
  const reports: (MicrophoneProcessingReport | null)[] = [];
  const processing = new MicrophoneProcessing(report => reports.push(report));
  await processing.configure(microphone());
  expect(reports.at(-1)).toEqual({ callAudioRouting: "unavailable", echoCancellation: null, noiseSuppression: null, autoGainControl: null, configurationFailed: false });
  await processing.configure(microphone({ echoCancellation: true }, async () => { throw new Error("Unsupported changes"); }));
  expect(reports.at(-1)).toEqual({ callAudioRouting: "unavailable", echoCancellation: true, noiseSuppression: null, autoGainControl: null, configurationFailed: true });
  processing.close();
});

test("late settings from a replaced microphone or closed call cannot overwrite the current report", async () => {
  const reports: (MicrophoneProcessingReport | null)[] = [];
  let finish: (() => void) | undefined;
  const pending = new Promise<void>(resolve => { finish = resolve; });
  const processing = new MicrophoneProcessing(report => reports.push(report));
  const old = processing.configure(microphone({ echoCancellation: false }, () => pending));
  await processing.configure(microphone({ echoCancellation: true }));
  finish?.(); await old;
  expect(reports).toHaveLength(1);
  expect(reports[0]?.echoCancellation).toBe(true);
  const closing = new MicrophoneProcessing(report => reports.push(report));
  const late = closing.configure(microphone());
  closing.close(); await late;
  expect(reports).toHaveLength(1);
  processing.close();
});

test("microphone warnings distinguish disabled, unreported, and rejected processing", () => {
  const view = new MicrophoneStatus();
  expect(view.element.hidden).toBe(true);
  const report = { callAudioRouting: "unavailable", echoCancellation: false, noiseSuppression: true, autoGainControl: null, configurationFailed: false } as const;
  view.update(report);
  expect(view.element.textContent).toContain("echo cancellation is off");
  view.update({ ...report, echoCancellation: null });
  expect(view.element.textContent).toContain("did not report echo cancellation");
  view.update({ ...report, echoCancellation: true, configurationFailed: true });
  expect(view.element.textContent).toContain("could not apply");
  expect(view.element.textContent).toContain("does not measure echo removal");
  view.update({ ...report, echoCancellation: true, callAudioRouting: "play-and-record" });
  expect(view.element.textContent).toContain("Browser reports echo cancellation on");
  expect(view.element.textContent).toContain("Speaker echo can still reach the microphone");
  expect(view.element.textContent).toContain("Call audio routing: Play and record");
  view.update({ ...report, callAudioRouting: "not-applied" });
  expect(view.element.textContent).toContain("Could not apply call audio routing");
  view.update(null);
  expect(view.element.hidden).toBe(true);
});
