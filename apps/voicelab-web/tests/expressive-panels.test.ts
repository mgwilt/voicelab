import { expect, test } from "bun:test";
import { createPanel as drama } from "../frontend/models/drama/panel.js";
import { createPanel as maya } from "../frontend/models/maya/panel.js";
import { createPanel as omni } from "../frontend/models/omnivoice/panel.js";
import { createPanel as kitten } from "../frontend/models/kitten/panel.js";

const base = { kind: "synthesis", label: "Test", description: "Test", backend: "Test", available: true, reason: null } as const;

test("expressive panels expose reference inputs and disable advanced controls during runs", () => {
  const panels = [
    drama({ ...base, id: "drama", endpoint: "/api/drama/session", outputSampleRate: 44100 }),
    maya({ ...base, id: "maya", endpoint: "/api/maya/session", outputSampleRate: 24000 }),
    omni({ ...base, id: "omnivoice", endpoint: "/api/omnivoice/session", outputSampleRate: 24000 }),
    kitten({ ...base, id: "kitten", endpoint: "/api/kitten/session", outputSampleRate: 24000 }),
  ];
  expect(panels[0]?.element.querySelectorAll('input[type="file"]').length).toBe(6);
  for (const panel of panels) {
    expect(panel.element.querySelector('input[type="file"]')).not.toBeNull();
    const advanced = [...panel.element.querySelectorAll("details")].find(section => section.querySelector("summary")?.textContent === "Advanced generation");
    expect(advanced?.open).toBe(false);
    panel.setDisabled(true);
    for (const input of panel.element.querySelectorAll("input,textarea,select")) {
      expect(input.hasAttribute("disabled")).toBe(true);
    }
    panel.setDisabled(false);
    for (const input of panel.element.querySelectorAll("input,textarea,select")) {
      expect(input.hasAttribute("disabled")).toBe(false);
    }
    panel.dispose();
  }
});
