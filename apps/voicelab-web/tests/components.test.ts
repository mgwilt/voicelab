import { expect, test } from "bun:test";
import { Lab } from "../frontend/components/lab.js";
import { Measurements } from "../frontend/components/measurements.js";
import { ModelPicker } from "../frontend/components/model-picker.js";
import type { ControlPanel } from "../frontend/components/panel.js";
import type { Emit, Model, Outcome } from "../frontend/domain/experiment.js";

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
  button(lab.element, "Start test").click();
  const next = models[1];
  if (next === undefined) throw new Error("Missing model");
  await lab.select(next);
  expect(order).toEqual(["start:nemotron", "cancel:nemotron", "dispose:nemotron"]);
  expect(button(lab.element, "Generate speech").disabled).toBe(false);
  await lab.dispose();
});
