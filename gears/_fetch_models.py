from pathlib import Path

import requests

MODELS_DIR = Path(__file__).parent / "data" / "session_models"
RELEASE_TAG = "models-v1"
REPO = "yvenn-amara/GEARS"
FILES = [
    "gmm_french.joblib",
    "gmm_french_holdout.joblib",
    "gmm_french_recency.joblib",
    "gmm_french_sample.joblib",
    "vae_french_sample.joblib",
]

def ensure_models():
    MODELS_DIR.mkdir(exist_ok=True)
    for name in FILES:
        dest = MODELS_DIR / name
        if not dest.exists():
            url = f"https://github.com/{REPO}/releases/download/{RELEASE_TAG}/{name}"
            r = requests.get(url)
            r.raise_for_status()
            dest.write_bytes(r.content)
