# Kokoro

Open **Direct run** for the synthesis workflows below. Use [Conversation](../../usage/conversations.md) for ongoing voice calls.

Kokoro 82M generates 24 kHz speech with 54 preset voices. The lab uses MLX-Audio
and the upstream Misaki text frontend.

## Run

```sh
./voicelab run kokoro --text 'The price is twelve dollars.' --voice af_heart
./voicelab run kokoro --text 'Bonjour le monde.' --voice ff_siwis
```

Voice prefixes select American English (`a`), British English (`b`), Spanish
(`e`), French (`f`), Hindi (`h`), Italian (`i`), Japanese (`j`), Portuguese (`p`),
or Mandarin (`z`). The website lists all 54 voices. `--lang-code` can explicitly
confirm the selected language. Mismatches fail before loading.

Use comma-separated IDs to blend voices of the same language, for example
`--voice af_heart,af_bella`. `--speed` accepts 0.5–2.0.
The released model has no voice encoder; reference-audio cloning is not supported.

## Frontend and measurements

Misaki retains its upstream lexicons, number handling, spaCy part-of-speech tags,
and language-specific phonemizers. The prior English eSpeak-only substitute has
been removed. English uses the upstream default small spaCy model.
The vendored Misaki source and dictionaries are unchanged; local dependency
metadata permits the tested Python version. See `third_party/misaki/UPSTREAM.md`.

Japanese initializes the official UniDic 3.1 dictionary on first use. Its download
is checksum verified and stored in the local user cache. Frontend initialization
occurs before generation timing; `preparation_seconds` records that work.
The runtime emits decoded text segments and predicts their duration directly.
Completion does not require a sampled EOS token.

All nine frontend paths passed local phonemization and synthesis smoke checks.
Local ASR recognized the test content, with transliteration and minor spelling
differences in some languages. These checks do not establish pronunciation or
voice quality across every language or preset.
This runtime requires Apple Silicon macOS. Checkpoint tensors are float32 despite
the repository's `bf16` name. See [measurements](../../usage/measurements.md).

## Provenance

- [Model](https://huggingface.co/hexgrad/Kokoro-82M): `f3ff3571791e39611d31c381e3a41a3af07b4987`.
- [MLX checkpoint](https://huggingface.co/mlx-community/Kokoro-82M-bf16): `a71e4d38b236d968966a2002c4c895dbd12b1c3c`.
- [Misaki](https://github.com/hexgrad/misaki): official 0.9.4 wheel, SHA-256 recorded beside the vendored source.
- UniDic: official 3.1.0 archive, SHA-256 `638718c4c63625ab300de4c92c67925d54c0e9e3830009eaa992f29819d59c43`.

Model and Misaki retain Apache-2.0 terms. The dictionary cache retains UniDic's
upstream licenses. The lab does not distribute model or dictionary weights.
