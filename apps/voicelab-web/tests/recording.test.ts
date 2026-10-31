import { expect, test } from "bun:test";
import { MAX_RECORDING_BYTES, recordingWav, type RecordingCapture } from "../frontend/audio/recording.js";
import { AudioInput } from "../frontend/components/audio-input.js";

class FakeCapture implements RecordingCapture {
  receive: (pcm: ArrayBuffer) => void = () => {};
  signal: AbortSignal | null = null;
  closed = 0;
  resolve: (() => void) | null = null;
  fail = false;
  async open(signal: AbortSignal, receive: (pcm: ArrayBuffer) => void): Promise<void> {
    this.signal = signal; this.receive = receive;
    if (this.fail) throw new Error("Permission denied");
    await new Promise<void>(resolve => { this.resolve = resolve; });
  }
  async close(): Promise<void> { this.closed++; }
}
function button(input: AudioInput, title: string): HTMLButtonElement {
  const result = [...input.element.querySelectorAll("button")].find(item => item.textContent === title);
  if (result === undefined) throw new Error(`Missing ${title}`);
  return result;
}
async function flush(): Promise<void> { await Promise.resolve(); await Promise.resolve(); await Promise.resolve(); }

test("recorded WAV has bounded mono 16 kHz PCM16 metadata and unchanged samples", async () => {
  const pcm = new Uint8Array([0, 128, 255, 127]);
  const file = recordingWav([pcm]);
  const bytes = await file.arrayBuffer(); const view = new DataView(bytes);
  expect(file.type).toBe("audio/wav"); expect(file.size).toBe(48);
  expect(new TextDecoder().decode(bytes.slice(0, 4))).toBe("RIFF");
  expect(view.getUint16(20, true)).toBe(1); expect(view.getUint16(22, true)).toBe(1);
  expect(view.getUint32(24, true)).toBe(16000); expect(view.getUint16(34, true)).toBe(16);
  expect(view.getUint32(40, true)).toBe(4); expect(new Uint8Array(bytes.slice(44))).toEqual(pcm);
  expect(() => recordingWav([])).toThrow();
  expect(() => recordingWav([new Uint8Array(3)])).toThrow();
  expect(() => recordingWav([new Uint8Array(MAX_RECORDING_BYTES + 2)])).toThrow();
});

test("recording needs an explicit start and stop, then returns WAV and releases capture", async () => {
  const capture = new FakeCapture(); const input = new AudioInput("Question", () => capture);
  expect(input.read()).toBeNull(); expect(capture.signal).toBeNull();
  button(input, "Record audio").click(); expect(() => input.read()).toThrow("Stop recording");
  capture.resolve?.(); await flush(); capture.receive(new ArrayBuffer(2560));
  button(input, "Stop recording").click(); await flush();
  expect(capture.closed).toBe(1); expect(capture.signal?.aborted).toBe(true);
  expect(input.read()?.size).toBe(2604);
  capture.receive(new ArrayBuffer(2560)); expect(input.read()?.size).toBe(2604);
  button(input, "Clear audio").click(); expect(input.read()).toBeNull(); input.dispose();
});

test("30-second PCM bound truncates excess, stops capture, and excludes late callbacks", async () => {
  const capture = new FakeCapture(); const input = new AudioInput("Question", () => capture);
  button(input, "Record audio").click(); capture.resolve?.(); await flush();
  capture.receive(new ArrayBuffer(MAX_RECORDING_BYTES + 2560)); await flush();
  expect(input.read()?.size).toBe(MAX_RECORDING_BYTES + 44); expect(capture.closed).toBe(1);
  capture.receive(new ArrayBuffer(2560)); expect(input.read()?.size).toBe(MAX_RECORDING_BYTES + 44);
  input.dispose();
});

test("disable aborts pending permission; late completion cannot start or retain a recording", async () => {
  const capture = new FakeCapture(); const input = new AudioInput("Question", () => capture);
  button(input, "Record audio").click(); input.setDisabled(true); await flush();
  expect(capture.closed).toBe(1); expect(capture.signal?.aborted).toBe(true);
  capture.resolve?.(); await flush(); capture.receive(new ArrayBuffer(2560));
  expect(input.read()).toBeNull(); expect(button(input, "Record audio").disabled).toBe(true);
  input.setDisabled(false); expect(button(input, "Record audio").disabled).toBe(false);
  expect(button(input, "Stop recording").disabled).toBe(true); input.dispose();
});

test("dispose aborts recording, releases capture, and detaches action listeners", async () => {
  const capture = new FakeCapture(); const input = new AudioInput("Question", () => capture);
  const record = button(input, "Record audio"); record.click(); capture.resolve?.(); await flush();
  capture.receive(new ArrayBuffer(2560)); input.dispose(); await flush();
  expect(capture.closed).toBe(1); expect(capture.signal?.aborted).toBe(true);
  capture.receive(new ArrayBuffer(2560)); record.click(); expect(input.read()).toBeNull();
});

test("permission failure closes resources and leaves recording retry available", async () => {
  const capture = new FakeCapture(); capture.fail = true;
  const input = new AudioInput("Question", () => capture);
  button(input, "Record audio").click(); await flush();
  expect(capture.closed).toBe(1); expect(input.element.textContent).toContain("Permission denied");
  expect(button(input, "Record audio").disabled).toBe(false); input.dispose();
});

test("a new recording excludes an older permission grant and its stale PCM", async () => {
  const first = new FakeCapture(); const second = new FakeCapture();
  const captures = [first, second];
  const input = new AudioInput("Question", () => {
    const capture = captures.shift();
    if (capture === undefined) throw new Error("Unexpected capture allocation");
    return capture;
  });
  button(input, "Record audio").click(); input.setDisabled(true); await flush();
  input.setDisabled(false); button(input, "Record audio").click();
  first.resolve?.(); await flush(); first.receive(new ArrayBuffer(2560));
  second.resolve?.(); await flush(); second.receive(new ArrayBuffer(5120));
  button(input, "Stop recording").click(); await flush();
  expect(input.read()?.size).toBe(5164); expect(first.closed).toBe(1); expect(second.closed).toBe(1);
  input.dispose();
});

test("upload remains available and clear removes the selected file", () => {
  const input = new AudioInput("Question");
  const upload = input.element.querySelector("input");
  if (upload === null) throw new Error("Missing upload input");
  const file = recordingWav([new Uint8Array(2560)]);
  const files = upload.files;
  if (files === null) throw new Error("Missing file list");
  Object.defineProperty(files, "0", { value: file, configurable: true });
  Object.defineProperty(files, "length", { value: 1, writable: true });
  upload.dispatchEvent(new Event("change"));
  expect(input.read()?.size).toBe(2604);
  button(input, "Clear audio").click(); expect(input.read()).toBeNull(); input.dispose();
});
