import type { Emit, ExperimentSession } from "../domain/experiment.js";

export interface ControlPanel {
  readonly element: HTMLElement;
  createSession(emit: Emit): ExperimentSession;
  setDisabled(disabled: boolean): void;
  dispose(): void;
}
