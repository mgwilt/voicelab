# Fish Audio Drama 3

Open **Direct run** for the synthesis workflows below. Use [Conversation](../../usage/conversations.md) for ongoing voice calls.

Drama 3 generates expressive speech and clones voices from reference recordings.
Voice Lab runs the native MLX-Audio model locally and streams 44.1 kHz mono audio.

## Generation defaults

Temperature `1.0`, top-p `0.9`, and top-k `30` follow the author's
[published benchmark settings](https://huggingface.co/fishaudio/drama-3#performance).
This is a documented baseline, not a claim that those values are best for every voice.
Clearing the website's top-k field delegates filtering to the locked MLX runtime, which disables top-k by default.

## Generate speech

Use Apple Silicon macOS with enough memory for weights and inference caches.
The first run downloads approximately 23.8 GB, including the speech codec.

```sh
./voicelab run drama --runtime mlx --text '[laughs] Welcome to Voice Lab.'
```

Speaker tags use `<|speaker:0|>` through `<|speaker:15|>`.
Expression tags remain in the text. The adapter rejects reserved structural tokens.

## Use reference voices

```sh
./voicelab run drama --runtime mlx --text 'Welcome to Voice Lab.' \
  --reference-audio reference.wav --reference-text 'The words in this recording.'
```

Repeat both reference flags for up to four paired recordings.
Use mono PCM16 WAV files, 8–48 kHz, from 0.1 to 30 seconds each.
The website has the same paired audio and transcript controls.
Temporary reference files are removed after generation or cancellation.
Reports store audio hashes and metadata instead of recording data.

## Revise an earlier take

Supply earlier utterances separately from voice references.
Each earlier take contains its audio and exact transcript, in chronological order.
For example, save this manifest beside `wednesday.wav`:

```json
[{"audio": "wednesday.wav", "text": "The meeting is on Wednesday."}]
```

Then request the revised utterance:

```sh
./voicelab run drama --runtime mlx --text 'The meeting is on Thursday.' \
  --previous-takes takes.json
```

Add a second earlier take to condition a subsequent revision on both takes.
The website provides the same ordered audio and transcript inputs.
Use at most two earlier takes and four recordings total, including voice references.

This implements the model card's **editing as continuation** workflow.
The model generates a new complete utterance with earlier text and audio in its history.
It does not splice a recording or preserve unchanged samples.
The lab re-encodes uploaded WAVs with the pinned codec; it does not retain their original generation tokens.

## Adjust generation

Run `./voicelab run drama --help` for all options.
The website groups these options under **Advanced generation**.

- `--temperature`, `--top-k`, and `--top-p` control sampling.
- `--max-new-tokens` limits frames in each native text segment.
- `--streaming-interval` sets the target duration of decoded chunks.
- `--first-chunk-frames` changes the first decoded chunk size.
- `--seed` controls sampling. It does not promise identical results across devices.

The native chunker retains context between long-text segments.
Continuation history is supplied by the lab orchestration layer. Frame sampling and decoding use the locked native model.

## Read measurements

Real-time factor (RTF) includes reference conditioning, generation, decoding, synchronization, and callback delivery.
Loading is measured separately. First PCM precedes browser transport and playback.
Codec-only time remains null.

A sampled EOS stops each segment. Any segment that reaches its frame limit makes the run incomplete.
Cancellation closes the generator, resets codec state, and restores random state at a decoded chunk boundary.
Loaded models remain resident until `close()`.

Local runs exercised plain speech, single and multiple references, chained continuation revisions, completion, caps, cancellation, and reuse.
Real CLI and shared website transport tests generated a revised take from uploaded history.
A no-history comparison produced identical PCM to the locked native generator for the same request.
Conditioned PCM differed from fixed-seed baseline PCM. This does not establish speaker similarity or audio quality.
CUDA, TPU, and ANE execution remain unavailable in this adapter.

## Provenance

Source: [fishaudio/drama-3](https://huggingface.co/fishaudio/drama-3), revision
`78e5ab79e49aa9a25abe8799eff2c4d73a1c3216`.
Conversion: [mlx-community/fish-audio-drama-3-4bit](https://huggingface.co/mlx-community/fish-audio-drama-3-4bit), revision
`1ac1b2108b63c2ef5bade463a121569363e293ce`.
The conversion uses 4-bit experts, 8-bit projections, and BF16 remaining weights.
The pinned snapshot includes the codec. Runtime: `mlx-audio==0.6.3`, native `fish_drama3` implementation.

Weights retain the Fish Audio model license. MLX-Audio uses MIT terms.
The adapted orchestration retains its MIT [LICENSE](../../../packages/voicelab-inference/src/voicelab_inference/drama/mlx/LICENSE) and [NOTICE](../../../packages/voicelab-inference/src/voicelab_inference/drama/mlx/NOTICE).
The lab does not distribute weights or claim numerical parity with the source runtime.
