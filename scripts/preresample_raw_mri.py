"""Pre-resample raw MRI volumes to 1×1×3 mm and cache them under data/preprocessed.

Run once before training with mri_type=raw to eliminate per-sample resampling
overhead in the DataLoader.

Usage:
    python scripts/preresample_raw_mri.py [--workers N] [--force]
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np

import hcc_multimodal
from hcc_multimodal.contrastive.transform import resample_to_spacing

_DATA_ROOT = Path(hcc_multimodal.__file__).parent.parent / "data"
_MRI_ROOT_RAW = _DATA_ROOT / "Resection" / "Images" / "Resections_with_rna"
_MRI_ROOT_CACHE = _DATA_ROOT / "mri_resampled"
_FILENAME = "MRI_liver_arterial.nii.gz"


def _resample_patient(src: Path, dst: Path, force: bool) -> tuple[int, str]:
    pid = int(src.parent.name)
    if dst.exists() and not force:
        return pid, "skip"
    img = nib.load(src)
    vol = resample_to_spacing(img)
    new_img = nib.Nifti1Image(vol, affine=np.eye(4))
    dst.parent.mkdir(parents=True, exist_ok=True)
    nib.save(new_img, dst)
    return pid, "done"


def main() -> None:
    p = argparse.ArgumentParser(description="Pre-resample raw MRI to 1×1×3 mm cache")
    p.add_argument("--workers", type=int, default=4, help="Parallel worker processes")
    p.add_argument("--force", action="store_true", help="Overwrite existing cached files")
    args = p.parse_args()

    sources = sorted(_MRI_ROOT_RAW.glob(f"*/{_FILENAME}"))
    if not sources:
        sys.exit(f"No raw MRI files found under {_MRI_ROOT_RAW}")

    tasks = [
        (src, _MRI_ROOT_CACHE / src.parent.name / _FILENAME, args.force)
        for src in sources
    ]

    done = skipped = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_resample_patient, *t): t[0] for t in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            pid, status = fut.result()
            if status == "done":
                done += 1
            else:
                skipped += 1
            print(f"[{i}/{len(tasks)}] {pid}: {status}", flush=True)

    print(f"\nFinished — {done} resampled, {skipped} skipped. Cache: {_MRI_ROOT_CACHE}")


if __name__ == "__main__":
    main()
