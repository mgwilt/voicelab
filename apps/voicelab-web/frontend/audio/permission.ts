interface StreamTracks { getTracks(): readonly { stop(): void }[] }
/** A late permission grant must release the device after its session was cancelled. */
export async function acquireStream<T extends StreamTracks>(
  pending: Promise<T>, signal: AbortSignal,
): Promise<T> {
  return new Promise((resolve, reject) => {
    const abort = (): void => reject(new DOMException("Microphone cancelled", "AbortError"));
    signal.addEventListener("abort", abort, { once: true });
    pending.then(stream => {
      signal.removeEventListener("abort", abort);
      if (signal.aborted) {
        stream.getTracks().forEach(track => track.stop());
        abort();
      } else resolve(stream);
    }, error => {
      signal.removeEventListener("abort", abort);
      reject(error);
    });
    if (signal.aborted) abort();
  });
}
