import { element } from "./dom.js";

type WorkbenchView = "controls" | "output";
let nextId = 0;

/** View navigation owns visibility only; panels retain their DOM and session state. */
export class Workbench {
  readonly element = element("div", "workbench");
  private readonly listeners = new AbortController();
  private readonly views = new Map<WorkbenchView, { button: HTMLButtonElement; panel: HTMLElement }>();

  constructor(controls: HTMLElement, output: HTMLElement) {
    const id = `workbench-${++nextId}`;
    const tabs = element("div", "workbench-tabs");
    tabs.setAttribute("role", "tablist");
    tabs.setAttribute("aria-label", "Session views");
    const entries: readonly [WorkbenchView, string, HTMLElement][] = [
      ["controls", "Controls", controls], ["output", "Output", output],
    ];
    for (const [view, label, panel] of entries) {
      const button = element("button", "workbench-tab", label);
      button.type = "button";
      button.id = `${id}-${view}-tab`;
      button.setAttribute("role", "tab");
      button.setAttribute("aria-controls", `${id}-${view}`);
      panel.id = `${id}-${view}`;
      panel.setAttribute("role", "tabpanel");
      panel.setAttribute("aria-labelledby", button.id);
      panel.tabIndex = 0;
      button.addEventListener("click", () => this.show(view), { signal: this.listeners.signal });
      button.addEventListener("keydown", event => {
        if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
        event.preventDefault();
        const next = event.key === "Home" ? "controls" : event.key === "End" ? "output"
          : view === "controls" ? "output" : "controls";
        this.show(next);
        this.views.get(next)?.button.focus();
      }, { signal: this.listeners.signal });
      this.views.set(view, { button, panel });
      tabs.append(button);
    }
    this.element.append(tabs, controls, output);
    this.show("controls");
  }

  show(view: WorkbenchView): void {
    for (const [key, { button, panel }] of this.views) {
      const selected = key === view;
      button.setAttribute("aria-selected", String(selected));
      button.tabIndex = selected ? 0 : -1;
      panel.hidden = !selected;
    }
  }

  dispose(): void { this.listeners.abort(); }
}
