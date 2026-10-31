import { expect, test } from "bun:test";
import { Workbench } from "../frontend/components/workbench.js";

test("tabs expose associated panels and support keyboard activation without moving content", () => {
  const controls = document.createElement("section");
  const output = document.createElement("section");
  const workbench = new Workbench(controls, output);
  document.body.append(workbench.element);
  try {
    const tabs = [...workbench.element.querySelectorAll("button")];
    const first = tabs[0]; const second = tabs[1];
    if (first === undefined || second === undefined) throw new Error("Missing tabs");
    expect(first.getAttribute("aria-controls")).toBe(controls.id);
    expect(output.getAttribute("aria-labelledby")).toBe(second.id);
    expect(first.getAttribute("aria-selected")).toBe("true");
    expect(first.tabIndex).toBe(0);
    expect(second.tabIndex).toBe(-1);
    expect(output.hidden).toBe(true);
    first.focus();
    first.dispatchEvent(new KeyboardEvent("keydown", { key: "ArrowRight", bubbles: true }));
    expect(document.activeElement).toBe(second);
    expect(output.hidden).toBe(false);
    expect(controls.hidden).toBe(true);
    expect(second.getAttribute("aria-selected")).toBe("true");
    second.dispatchEvent(new KeyboardEvent("keydown", { key: "Home", bubbles: true }));
    expect(document.activeElement).toBe(first);
    expect(controls.hidden).toBe(false);
    workbench.dispose();
    second.click();
    expect(output.hidden).toBe(true);
  } finally {
    workbench.dispose();
    workbench.element.remove();
  }
});
