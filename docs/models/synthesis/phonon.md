# Phonon

Open **Direct run** for the synthesis workflows below. Use [Conversation](../../usage/conversations.md) for ongoing voice calls.

Phonon 31M generates English speech at 24 kHz. Voice Lab uses its native
Transformers model and a frame-by-frame inference loop.

## Run

```sh
./voicelab run phonon --text 'Hello from Voice Lab.' --voice Easton
```

Choose `default` or one of 37 released English voices. The shared website lists
all voices. Use `--temperature`, `--seed`, and `--max-frames` to control sampling.
Use `--no-streaming` to compare with native buffered Transformers generation.
`--device` accepts `cpu`, `mps`, or `cuda`; unavailable devices fail explicitly.

## Capabilities and limits

Streaming delivers one 80 ms frame while later frames are still generated.
The baseline decoder recomputes the causal prefix. This preserves output but
costs more than buffered decoding. CPU comparison matched the buffered waveform
within 7e-7 maximum absolute difference on a short test.
A frame limit marks incomplete output. Cancellation closes the frame iterator.

The released codec has no speaker encoder. Local raw-audio cloning is not
available from these weights; Gradium advertises cloning through its hosted
studio. Preset voices use released speaker latents.
This English checkpoint does not become multilingual by selecting another voice.
The drifting flow head uses one step; a flow-step knob would have no effect.

The native Transformers frontend does not expand numbers, currency, dates, or
addresses like the separate Phonon Rust runtime. Write the intended words for
those inputs. No text-normalization parity is claimed.

## Provenance

- [Model](https://huggingface.co/gradium/phonon-31m-en): `92741c5b07d085ff840e26fea8ce43110b87852f`.
- [Decoder](https://huggingface.co/gradium/phonon-mimi): `33519be1a8e61c6fa00c6e154e2d38e76a367919`.
- [Voice catalog](https://huggingface.co/gradium/phonon-voices): `0ee17205f254b5abc43b266c8a0f215565c352f7`.

Weights and adapted Transformers sampling retain Apache-2.0 terms. See the
runtime's LICENSE and NOTICE. Model and preset downloads occur during loading.
See [measurement rules](../../usage/measurements.md).
