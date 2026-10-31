/** Boundary primitives only; domain objects never circulate as untyped records. */
function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
export function record(value: unknown): Record<string, unknown> {
  if (!isRecord(value)) {
    throw new Error("Expected a JSON object");
  }
  return value;
}
export function parse(raw: string): Record<string, unknown> {
  const value: unknown = JSON.parse(raw);
  return record(value);
}
export function text(value: unknown): string {
  if (typeof value !== "string") throw new Error("Expected text");
  return value;
}
export function finite(value: unknown): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    throw new Error("Expected a finite nonnegative number");
  }
  return value;
}
export function nullableNumber(value: unknown): number | null {
  return value === null ? null : finite(value);
}
export function boolean(value: unknown): boolean {
  if (typeof value !== "boolean") throw new Error("Expected a boolean");
  return value;
}
