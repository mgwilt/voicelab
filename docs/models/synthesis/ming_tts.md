# Ming TTS

Open **Direct run** for the synthesis workflows below. Use [Conversation](../../usage/conversations.md) for ongoing voice calls.

Ming 3.0 TTS 0.5B generates 44.1 kHz speech with local voice cloning, built-in
voices, voice design, and delivery controls. The lab uses the dense MLX model.

## Generation defaults

Guidance `2.0`, sigma `0.25`, and temperature `0.0` match the official
[inference defaults](https://github.com/inclusionAI/Ming-3.0-tts/blob/9332fb9443325d4726ce02cefd4ec1c211df4079/infer.py#L76).
The lab retains the locked MLX runtime's `10` flow steps.
Temperature controls solver noise, not text-token sampling.

## Run

```sh
./voicelab run ming-tts --text 'Welcome to Voice Lab.' --voice US_Emma
./voicelab run ming-tts --text 'A new sentence in the reference voice.' \
  --ref-audio reference.wav --ref-text 'Exact words in the reference clip.'
./voicelab run ming-tts --text 'A quiet evening.' \
  --instruction 'Speak softly and unhurriedly.'
```

Reference audio and its exact transcript are required together. Use mono PCM16
WAV, 8–48 kHz, up to 30 seconds. Reference conditioning uses the released VAE and
CAM++ speaker encoder. Keep clips short enough to fit the model's context.
Trailing silence is trimmed only from acoustic conditioning by default: a -40 dB
relative amplitude threshold, 50 ms tail margin, and at least 400 ms retained.
The speaker encoder uses the original clip. Use `--no-trim-reference-silence` to
compare without trimming. Reports retain the original hash and this option.

The website lists the built-in voice catalog. Select `--language` from `zh`, `en`,
`ja`, `ko`, `de`, `fr`, or `es`. A language selection does not replace voice selection.

## Choose controls

- `--instruction`: freeform delivery instructions. Use alone or with a reference voice.
- `--style`: voice-design description. Do not combine with a built-in or reference voice.
- `--voice`: an exact built-in voice ID, such as `US_Emma`, `UK_George`, or `灵小柔`.
- `--emotion`: `高兴`, `悲伤`, `愤怒`, `恐惧`, `厌恶`, `惊讶`, or `中性`.
- `--dialect`: `cantonese` or `chuanyu dialect`.
- `--speed`: `慢速`, `中速`, or `快速`; `--pitch` and `--volume`: `低`, `中`, or `高`.

Freeform instructions cannot combine with structured voice, design, emotion,
dialect, speed, pitch, or volume controls. Invalid combinations fail before loading.
Inline tags such as `[Breathing]`, `[Sigh]`, and `[Laughter]` pass through in the text.
The [official cookbook](https://github.com/inclusionAI/Ming-3.0-tts/blob/9332fb9443325d4726ce02cefd4ec1c211df4079/cookbooks/instruction_en.md)
lists all tags and describes the exact control values.

Sampling controls are `--seed`, `--max-frames`, `--cfg-scale`, `--flow-steps`,
`--temperature`, and `--sigma`. Frame limits apply per text segment.

## Measurements and limits

The runtime decodes complete text segments. First PCM is segment availability,
not the first generated latent. RTF includes reference encoding, sampling,
decoding, and output delivery. Model loading is excluded.
The stop head determines completion. A frame or context limit remains incomplete.
Cancellation closes sampling when the next decoded segment reaches delivery.

The reference encoder uses the official Oobleck posterior: softplus of the scale
plus 1e-4. The bundled MLX implementation's log-variance interpretation is not used.
Speaker features follow the official CAM++ frontend and are computed locally on CPU.

This runtime preserves MLX-Audio's sentence splitting and period handling.
It does not implement the official optional Pynini FST normalization for numbers,
dates, or units. Write those inputs as intended words; normalization parity is not
claimed. This runtime requires Apple Silicon macOS.
Local smoke tests completed for reference cloning, Spanish speech from an English
reference, a built-in voice, and style with emotion. Local ASR recovered the
requested content. The shared FastAPI reference-upload route returned complete
PCM. These checks do not measure speaker similarity or subjective audio quality.
See [measurement rules](../../usage/measurements.md).

## Provenance

- [Source model](https://huggingface.co/inclusionAI/Ming-3.0-tts-0.5B): `7247f4c94a5b2e987404ed28a5cfec41b1a4ca61`.
- [8-bit MLX checkpoint](https://huggingface.co/mlx-community/Ming-3.0-tts-0.5B-8bit): `d0d1da6f627e4b3f20cc2db6ba4aee86d45343ad`.
- CAM++: `campplus.onnx` from the pinned source model.
- Official control and speaker frontend: source revision `9332fb9443325d4726ce02cefd4ec1c211df4079`.

Model and adapted MLX code retain their MIT terms. See runtime LICENSE and NOTICE.
