import type { Emit, ExperimentSession, Outcome, Phase, SessionEvent } from "./experiment.js";

export interface RunState {
  readonly phase: Phase;
  readonly message: string;
  readonly error: boolean;
}

/** One owner for async lifecycle; stale callbacks cannot update a newer run. */
export class SessionRunner {
  private current: ExperimentSession | null = null;
  private cancellation: AbortController | null = null;
  private pending: Promise<void> | null = null;
  private epoch = 0;
  private disposed = false;

  constructor(
    private readonly onState: (state: RunState) => void,
    private readonly onEvent: Emit,
  ) {}

  get busy(): boolean { return this.current !== null; }

  start(create: (emit: Emit) => ExperimentSession): void {
    if (this.disposed || this.busy) return;
    const epoch = ++this.epoch;
    const cancellation = new AbortController();
    const emit = (event: SessionEvent): void => {
      if (this.disposed || epoch !== this.epoch || cancellation.signal.aborted) return;
      if (event.type === "status") this.onState({ phase: event.phase, message: event.message, error: false });
      else this.onEvent(event);
    };
    try {
      this.current = create(emit);
    } catch (error) {
      this.onState({ phase: "idle", message: error instanceof Error ? error.message : String(error), error: true });
      return;
    }
    this.cancellation = cancellation;
    this.onState({ phase: "preparing", message: "Starting experiment…", error: false });
    this.pending = this.execute(this.current, cancellation, epoch);
  }

  private async execute(session: ExperimentSession, cancellation: AbortController, epoch: number): Promise<void> {
    let outcome: Outcome = { kind: "cancelled", message: "Session stopped" };
    let errorMessage: string | null = null;
    try {
      outcome = await session.run(cancellation.signal);
    } catch (error) {
      if (!cancellation.signal.aborted && !(error instanceof Error && error.name === "AbortError")) {
        errorMessage = error instanceof Error ? error.message : String(error);
      }
    } finally {
      if (epoch === this.epoch) {
        this.current = null;
        this.cancellation = null;
        this.pending = null;
        if (!this.disposed) {
          this.onState({ phase: "idle", message: errorMessage ?? outcome.message, error: errorMessage !== null });
        }
      }
    }
  }

  stop(): void { this.current?.stop(); }

  async cancel(): Promise<void> {
    this.cancellation?.abort();
    await this.pending;
  }

  async dispose(): Promise<void> {
    this.disposed = true;
    await this.cancel();
    this.epoch++;
  }
}
