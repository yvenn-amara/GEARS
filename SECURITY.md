# Security policy

## Supported versions

Fixes go into the latest 2.x release.

## Reporting a vulnerability

Please do not open a public issue. Use GitHub's private reporting: the **Security** tab of the
repository, then **Report a vulnerability**
(<https://github.com/EDF-Lab/GEARS/security/advisories/new>). Say which version you use, what you did,
and what happened. A maintainer will answer there.

## Model files are pickles

GEARS saves a fitted model with `joblib`, which is `pickle`: **opening a model file can run arbitrary code.**

- `load_pretrained("french")` downloads the official model and checks its SHA-256 against the value in
  `gears/pretrained.py` **before** opening it. The same check applies when you pass `path=`. The checksum of
  each release file is also published in `SHA256SUMS` on the release page.
- `EVSessionModel.load(path)` and `GEARSModel.load(path)` open the file you give them without any check.
  Open only files you saved yourself or got from a source you trust, and compare their `sha256sum` with the
  published one.
- `import gears` does not touch the network or the disk. The only download is `load_pretrained`, and
  `GEARS_OFFLINE=1` forbids it.

GEARS ships no data.
