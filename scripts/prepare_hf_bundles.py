"""
scripts/prepare_hf_bundles.py
─────────────────────────────
Prepare and optionally upload GEARS pre-trained model bundles to HF Hub.

Usage
-----
# Prepare demo bundles (synthetic data, no upload)
python scripts/prepare_hf_bundles.py --demo

# Prepare from real data files and upload
python scripts/prepare_hf_bundles.py \
    --data-dir data/real/ \
    --upload \
    --token hf_YOUR_TOKEN

# Prepare one specific model
python scripts/prepare_hf_bundles.py \
    --model-id work_fr_idf \
    --data-file data/real/work_idf.csv \
    --location-type work \
    --upload \
    --token hf_YOUR_TOKEN
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# Add project root to path so the script works from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))


def prepare_demo_bundles(cache_dir: Path) -> None:
    """Generate synthetic demo bundles for all three location types."""
    from gears.models.registry import _CATALOGUE, ModelRegistry

    registry = ModelRegistry(cache_dir=cache_dir)

    for model_id, meta in _CATALOGUE.items():
        if "demo" not in model_id:
            continue
        logger.info("Preparing demo bundle: %s", model_id)
        path = registry._generate_demo_bundle(
            model_id, cache_dir / f"{model_id}.joblib"
        )
        logger.info("  → saved to %s", path)


def prepare_from_data(
    model_id: str,
    data_file: Path,
    location_type: str,
    cache_dir: Path,
    n_components: int | str = "auto",
    random_state: int = 42,
) -> Path:
    """Fit a model on real data and save a bundle."""
    from gears.data.loader import load_sessions
    from gears.models.forecaster import SessionForecaster
    from gears.models.registry import ModelRegistry
    from gears.models.session_model import EVSessionModel

    logger.info("Loading data from %s …", data_file)
    df = load_sessions(data_file, verbose=True)

    logger.info("Fitting GMM …")
    gmm = EVSessionModel(n_components=n_components, random_state=random_state)
    gmm.fit(df)
    logger.info("  %s", gmm)

    logger.info("Fitting forecaster …")
    fc = SessionForecaster(method="probabilistic").fit(df)

    metadata = {
        "model_id": model_id,
        "location_type": location_type,
        "source_file": str(data_file),
        "n_sessions": len(df),
        "synthetic": False,
    }

    registry = ModelRegistry(cache_dir=cache_dir)
    path = registry.save_bundle(
        model_id=model_id, gmm=gmm, forecaster=fc, metadata=metadata
    )
    logger.info("Bundle saved: %s", path)
    return path


def upload_bundle(model_id: str, cache_dir: Path, token: str) -> None:
    from gears.models.registry import ModelRegistry

    registry = ModelRegistry(cache_dir=cache_dir)
    registry.upload_to_hub(model_id=model_id, token=token)
    logger.info("Uploaded '%s' to HF Hub.", model_id)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare and upload GEARS pre-trained model bundles."
    )
    parser.add_argument("--demo", action="store_true",
                        help="Generate synthetic demo bundles.")
    parser.add_argument("--data-file", type=Path, default=None,
                        help="Path to real data CSV/Parquet.")
    parser.add_argument("--model-id", default=None,
                        help="Registry model ID for the bundle.")
    parser.add_argument("--location-type", default="work",
                        choices=["work", "home", "public"],
                        help="Location type for the data.")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Directory of real data files (auto-discovers *.csv).")
    parser.add_argument("--cache-dir", type=Path,
                        default=Path.home() / ".cache" / "gears" / "models",
                        help="Local cache directory.")
    parser.add_argument("--upload", action="store_true",
                        help="Upload bundles to HF Hub after saving.")
    parser.add_argument("--token", default=None,
                        help="HF Hub write token.")
    args = parser.parse_args()

    args.cache_dir.mkdir(parents=True, exist_ok=True)

    if args.demo:
        prepare_demo_bundles(args.cache_dir)

    elif args.data_file and args.model_id:
        prepare_from_data(
            model_id=args.model_id,
            data_file=args.data_file,
            location_type=args.location_type,
            cache_dir=args.cache_dir,
        )
        if args.upload:
            upload_bundle(args.model_id, args.cache_dir, args.token)

    elif args.data_dir:
        for csv_file in sorted(args.data_dir.glob("*.csv")):
            model_id = csv_file.stem  # use filename as model_id
            prepare_from_data(
                model_id=model_id,
                data_file=csv_file,
                location_type=args.location_type,
                cache_dir=args.cache_dir,
            )
            if args.upload:
                upload_bundle(model_id, args.cache_dir, args.token)

    else:
        parser.print_help()
        sys.exit(1)

    logger.info("Done ✓")


if __name__ == "__main__":
    main()
