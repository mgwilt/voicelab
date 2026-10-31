/** Preserve fractional phase across worklet blocks; emit continuous 80 ms frames. */
export class PcmResampler {
  private readonly pending: number[] = [];
  private position = 0;
  private readonly frame = new Int16Array(1280);
  private used = 0;

  constructor(private readonly inputRate: number) {
    if (!Number.isFinite(inputRate) || inputRate <= 0) throw new Error("Invalid sample rate");
  }
  push(input: Float32Array, emit: (buffer: ArrayBuffer) => void): void {
    this.pending.push(...input);
    const ratio = this.inputRate / 16000;
    while (this.position + 1 < this.pending.length) {
      const index = Math.floor(this.position);
      const left = this.pending[index];
      const right = this.pending[index + 1];
      if (left === undefined || right === undefined) throw new Error("Missing resampler sample");
      const fraction = this.position - index;
      const value = left * (1 - fraction) + right * fraction;
      this.frame[this.used++] = Math.round(Math.max(-1, Math.min(1, value)) * 32767);
      this.position += ratio;
      if (this.used === 1280) {
        const buffer = new ArrayBuffer(2560);
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
