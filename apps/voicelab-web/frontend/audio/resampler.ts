/** Preserve fractional phase across worklet blocks; emit continuous 80 ms frames. */
export class PcmResampler {
  private readonly pending: number[] = [];
  private position = 0;
  private readonly frame: Int16Array;
  private used = 0;

  constructor(private readonly inputRate: number, private readonly outputRate: 16000 | 24000 = 16000) {
    this.frame = new Int16Array(outputRate * 0.08);
    if (!Number.isFinite(inputRate) || inputRate <= 0) throw new Error("Invalid sample rate");
  }
  push(input: Float32Array, emit: (buffer: ArrayBuffer) => void): void {
    this.pending.push(...input);
    const ratio = this.inputRate / this.outputRate;
    while (this.position + 1 < this.pending.length) {
      const index = Math.floor(this.position);
      const left = this.pending[index];
      const right = this.pending[index + 1];
      if (left === undefined || right === undefined) throw new Error("Missing resampler sample");
      const fraction = this.position - index;
      const value = left * (1 - fraction) + right * fraction;
      this.frame[this.used++] = Math.round(Math.max(-1, Math.min(1, value)) * 32767);
      this.position += ratio;
      if (this.used === this.frame.length) {
        const buffer = new ArrayBuffer(this.frame.length * 2);
        const view = new DataView(buffer);
        this.frame.forEach((sample, i) => view.setInt16(i * 2, sample, true));
        emit(buffer);
        this.used = 0;
      }
    }
    const consumed = Math.min(Math.floor(this.position), this.pending.length);
    this.pending.splice(0, consumed);
    this.position -= consumed;
  }
}
