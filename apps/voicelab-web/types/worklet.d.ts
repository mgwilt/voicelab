declare const sampleRate: number;
declare class AudioWorkletProcessor { constructor(options?: AudioWorkletNodeOptions); readonly port: MessagePort; }
declare function registerProcessor(name: string, processor: typeof AudioWorkletProcessor): void;
