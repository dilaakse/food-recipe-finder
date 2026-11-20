"""Download the large weight files from Google Drive (they are not stored in git)."""
from . import config


def ensure_weights():
    """Download every missing file listed in config.WEIGHT_FILES into config.WEIGHTS_DIR."""
    config.WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    for fname, file_id in config.WEIGHT_FILES.items():
        target = config.WEIGHTS_DIR / fname
        if target.exists():
            continue
        if file_id.startswith("<"):
            raise RuntimeError(f"No Google Drive id set for {fname} in src/config.py (see README, 'Weights').")
        import gdown

        print(f"downloading {fname} ...")
        gdown.download(id=file_id, output=str(target), quiet=False)
        if not target.exists():
            raise RuntimeError(f"Download of {fname} failed. Check the sharing settings of the Drive file.")


if __name__ == "__main__":
    ensure_weights()
