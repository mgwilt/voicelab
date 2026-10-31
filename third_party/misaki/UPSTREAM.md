Misaki 0.9.4, copied unchanged from the official PyPI wheel.

Source: https://github.com/hexgrad/misaki
Wheel SHA-256: 90e2eeb169786c014c429e5058d2ea6bcd02d651f2a24450ba6c9ffc0f8da15a

Local packaging widens the Python upper bound for the tested Voice Lab environment. Runtime source and dictionary data are unchanged. Optional language dependencies are declared separately; English uses the non-transformer spaCy model.

The local `pyproject.toml` replaces wheel metadata. It declares only the English,
Japanese, and Mandarin extras used by Kokoro. English uses `phonemizer==3.4.0`
(which supplies `EspeakWrapper.set_data_path`) instead of `phonemizer-fork`.
The unused transformer spaCy backend is not installed; the application uses the
upstream default `trf=False` with the pinned `en_core_web_sm` package.
Japanese keeps the upstream Cutlet frontend and UniDic 3.1 dictionary.
The application wrapper remains subject to all Voice Lab coding checks.
