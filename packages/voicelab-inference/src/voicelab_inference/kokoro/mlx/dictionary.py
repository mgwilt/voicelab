"""Download and verify the pinned official UniDic dictionary."""

import hashlib
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

DICTIONARY_URL = "https://cotonoha-dic.s3-ap-northeast-1.amazonaws.com/unidic-3.1.0.zip"
DICTIONARY_SHA256 = "638718c4c63625ab300de4c92c67925d54c0e9e3830009eaa992f29819d59c43"


def dictionary_path() -> Path:
    root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "voicelab"
    destination = root / "unidic-3.1.0"
    marker = destination / "verified-sha256"
    if marker.is_file() and marker.read_text() == DICTIONARY_SHA256:
        return destination
    root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="unidic-", dir=root) as temporary:
        directory = Path(temporary)
        archive_path = directory / "dictionary.zip"
        digest = hashlib.sha256()
        with (
            urllib.request.urlopen(DICTIONARY_URL, timeout=60) as response,
            archive_path.open("wb") as output,
        ):
            while block := response.read(1024 * 1024):
                digest.update(block)
                output.write(block)
        if digest.hexdigest() != DICTIONARY_SHA256:
            raise ValueError(
                "UniDic dictionary checksum does not match the pinned release"
            )
        staged = directory / "unidic"
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                path = Path(member.filename)
                if (
                    path.is_absolute()
                    or ".." in path.parts
                    or path.parts[0] != "unidic"
                ):
                    raise ValueError("Invalid path in UniDic archive")
                if member.is_dir():
                    continue
                target = directory / path
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
        (staged / "mecabrc").write_text("# UniDic 3.1.0 for Voice Lab\n")
        (staged / "verified-sha256").write_text(DICTIONARY_SHA256)
        try:
            staged.rename(destination)
        except FileExistsError:
            if not marker.is_file() or marker.read_text() != DICTIONARY_SHA256:
                raise RuntimeError(
                    "An incomplete UniDic cache already exists"
                ) from None
    return destination
