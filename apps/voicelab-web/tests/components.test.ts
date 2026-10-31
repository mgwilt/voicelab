import { expect, test } from "bun:test";
import { Lab } from "../frontend/components/lab.js";
import { Measurements } from "../frontend/components/measurements.js";
import { ModelPicker } from "../frontend/components/model-picker.js";
import type { ControlPanel } from "../frontend/components/panel.js";
import type { Emit, Model, Outcome } from "../frontend/domain/session.js";

const models: readonly Model[] = [
  { id: "nemotron", kind: "duplex", label: "Nemotron", description: "Duplex", backend: "MLX", available: true, reason: null,
    endpoint: "/api/nemotron/session", inputSampleRate: 16000, outputSampleRate: 22050 },
  { id: "sesame", kind: "synthesis", label: "Sesame", description: "TTS", backend: "PyTorch", available: true, reason: null,
    endpoint: "/api/sesame/session", outputSampleRate: 24000 },
  { id: "ming_audio", kind: "conversation", label: "Ming Audio", description: "Audio and text tasks", backend: "MLX", available: true, reason: null,
    endpoint: "/api/ming_audio/session", outputSampleRate: 44100 },
];

function button(root: HTMLElement, label: string): HTMLButtonElement {
  const result = [...root.querySelectorAll("button")].find(candidate => candidate.textContent === label);
  if (result === undefined) throw new Error(`Missing button: ${label}`);
  return result;
}

test("model picker uses accessible selected state and releases handlers", () => {
  const selected: string[] = [];
  const picker = new ModelPicker(models, model => selected.push(model.id));
  expect([...picker.element.querySelectorAll(".tag")].map(tag => tag.textContent)).toEqual(["Live conversation", "Text to speech", "Audio and text"]);
  const first = picker.element.querySelector("button");
  if (first === null) throw new Error("Missing model card");
  picker.select("nemotron");
  expect(first.getAttribute("aria-pressed")).toBe("true");
  first.click();
  expect(selected).toEqual(["nemotron"]);
  picker.dispose();
  first.click();
  expect(selected).toEqual(["nemotron"]);
});

test("metrics distinguish the measurement denominator and reset on switch", () => {
  const metrics = new Measurements();
  metrics.update({ basis: "output", rtf: .9, audioSeconds: 3, latencyMs: 300, latencyLabel: "First PCM", backlogSeconds: null });
  expect(metrics.element.textContent).toContain("0.90×");
  expect(metrics.element.textContent).toContain("Output RTF");
  metrics.reset();
  expect(metrics.element.textContent).not.toContain("0.90×");
});

test("switching a running model cancels it before disposing its panel", async () => {
  const order: string[] = [];
  const create = (model: Model): ControlPanel => ({
    element: document.createElement("div"), setDisabled: () => {},
    dispose: () => { order.push(`dispose:${model.id}`); },
    createSession: (_emit: Emit) => ({
      run: (signal: AbortSignal) => new Promise<Outcome>((_resolve, reject) => {
        order.push(`start:${model.id}`);
        signal.addEventListener("abort", () => { order.push(`cancel:${model.id}`); reject(new DOMException("cancelled", "AbortError")); });
      }),
      stop: () => {},
    }),
  });
  const lab = new Lab(models, create);
  await Promise.resolve();
  await Promise.resolve();
  button(lab.element, "Start session").click();
  const next = models[1];
  if (next === undefined) throw new Error("Missing model");
  await lab.select(next);
  expect(order).toEqual(["start:nemotron", "cancel:nemotron", "dispose:nemotron"]);
  expect(button(lab.element, "Generate speech").disabled).toBe(false);
  await lab.dispose();
});

test("model explorer filters by workflow and keeps unavailable reasons discoverable", () => {
  const unavailable: Model = {
    id: "sesame", kind: "synthesis", label: "Sesame", description: "TTS", backend: "PyTorch",
    available: false, reason: "Install the runtime to enable this model.",
    endpoint: "/api/sesame/session", outputSampleRate: 24000,
  };
  const picker = new ModelPicker([...models.filter(model => model.id !== "sesame"), unavailable], () => {});
  const search = picker.element.querySelector("input");
  if (search === null) throw new Error("Missing model search");
  search.value = "text to speech";
  search.dispatchEvent(new Event("input"));
  const visible = [...picker.element.querySelectorAll("li")].filter(row => !row.hidden);
  expect(visible).toHaveLength(1);
  expect(visible[0]?.textContent).toContain("Install the runtime");
  expect(visible[0]?.querySelector("button")?.disabled).toBe(true);
  search.value = "not a model";
  search.dispatchEvent(new Event("input"));
  expect(picker.element.querySelector(".explorer-count")?.textContent).toBe("0 models");
  const empty = picker.element.querySelector<HTMLElement>(".explorer-content > .field-hint");
  expect(empty?.hidden).toBe(false);
  picker.dispose();
});

test("model explorer closes with Escape and returns keyboard focus to its disclosure", () => {
  const picker = new ModelPicker(models, () => {});
  document.body.append(picker.element);
  try {
    picker.element.open = true;
    const search = picker.element.querySelector("input");
    if (search === null) throw new Error("Missing model search");
    search.focus();
    search.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
    expect(picker.element.open).toBe(false);
    expect(document.activeElement).toBe(picker.element.querySelector("summary"));
  } finally {
    picker.dispose();
    picker.element.remove();
  }
});

test("workbench views preserve edits and active sessions while exposing output and stop", async () => {
  let emit: Emit = () => {};
  let stopCount = 0;
  let disposedCount = 0;
  const input = document.createElement("textarea");
  const lab = new Lab(models, () => ({
    element: input, setDisabled: disabled => { input.disabled = disabled; },
    dispose: () => { disposedCount++; },
    createSession: onEvent => {
      emit = onEvent;
      return {
        run: signal => new Promise<Outcome>((_resolve, reject) => {
          signal.addEventListener("abort", () => reject(new DOMException("cancelled", "AbortError")));
        }),
        stop: () => { stopCount++; },
      };
    },
  }));
  await Promise.resolve();
  await Promise.resolve();
  input.value = "Keep this prompt";
  button(lab.element, "Output").click();
  expect(lab.element.querySelector(".controls-view")?.hasAttribute("hidden")).toBe(true);
  button(lab.element, "Controls").click();
  expect(input.value).toBe("Keep this prompt");
  button(lab.element, "Start session").click();
  emit({ type: "measurement", value: { basis: "input", rtf: .75, audioSeconds: 4, latencyMs: 200, latencyLabel: "Frame p95", backlogSeconds: .1 } });
  emit({ type: "transcript", speaker: "assistant", text: "A live reply" });
  button(lab.element, "View output: Input RTF 0.75×").click();
  expect(lab.element.querySelector(".output-view")?.hasAttribute("hidden")).toBe(false);
  expect(lab.element.querySelector(".transcript")?.textContent).toContain("A live reply");
  expect(button(lab.element, "Start session").disabled).toBe(true);
  button(lab.element, "Stop").click();
  expect(stopCount).toBe(1);
  expect(disposedCount).toBe(0);
  expect(input.value).toBe("Keep this prompt");
  await lab.dispose();
  expect(disposedCount).toBe(1);
});
