import { element } from "./components/dom.js";
import { Application } from "./components/application.js";
import { createPanel } from "./models/panels.js";
import { loadCatalog } from "./transport/catalog.js";

const root = document.getElementById("app");
if (root === null) throw new Error("Missing application root");
const lifetime = new AbortController();
let lab: Application | null = null;

async function boot(target: HTMLElement): Promise<void> {
  target.replaceChildren(element("p", "loading", "Loading available models…"));
  try {
    const models = await loadCatalog(lifetime.signal);
    lifetime.signal.throwIfAborted();
    lab = new Application(models, createPanel);
    target.replaceChildren(lab.element);
  } catch (error) {
    if (lifetime.signal.aborted) return;
    const message = element("p", "load-error", error instanceof Error ? error.message : String(error));
    message.setAttribute("role", "alert");
    const retry = element("button", "button-secondary", "Try again");
    retry.type = "button";
    retry.addEventListener("click", () => { void boot(target); }, { once: true, signal: lifetime.signal });
    target.replaceChildren(message, retry);
  }
}
window.addEventListener("pagehide", () => {
  lifetime.abort();
  void lab?.dispose();
}, { once: true });
void boot(root);
