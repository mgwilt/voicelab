"""Interpret Phonon's end-of-speech threshold and decoder tail."""


def reached_eos(scores: tuple[float, ...], tail_frames: int) -> bool:
    """Match the pinned generation configuration, including a tail at the cap."""
    if tail_frames < 0:
        raise ValueError("EOS tail cannot be negative")
    return any(
        score > 4.0 and index + tail_frames < len(scores)
        for index, score in enumerate(scores)
    )
