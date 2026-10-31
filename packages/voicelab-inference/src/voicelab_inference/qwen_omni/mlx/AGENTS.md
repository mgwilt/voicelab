# Qwen Omni MLX

Read docs/models/conversation/qwen_omni.md before changing decoding.

- Preserve Thinker hidden states, rotary offsets, Talker conditioning, and codec frame alignment.
- Keep EOS evidence separate from generation caps. Check cancellation between decode steps.
- Measure first PCM before delivery. Keep preprocessing inside generation timing.
- Preserve NOTICE and LICENSE-mlx-vlm when adapting upstream code.
