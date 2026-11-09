"""
Download model weights into weights/ and clone third-party code into third_party/.

    python scripts/download_weights.py                 # everything: detection, calibration, 3D bodies
    python scripts/download_weights.py --no-3d         # skip the 3D body models
    python scripts/download_weights.py --list          # show what goes where

Files that are already present are skipped. See docs/getting-started/models.md
for the licence of each file and how to place them by hand.

The SAM 3D Body checkpoint is gated on Hugging Face: request access at
https://huggingface.co/facebook/sam-3d-body-dinov3 and run `hf auth login` first.
Until access is approved, the 3D view uses stand-in players; everything else works.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from var_offside.config.paths import project_home, weights_dir  # noqa: E402

# Upload best_tuned.pt as an asset of this GitHub release before publishing.
VAR_OFFSIDE_RELEASE = "https://github.com/Yousef-Nasr/var-offside/releases/download/weights-v0.1.0"
ULTRALYTICS_RELEASE = "https://github.com/ultralytics/assets/releases/download/v8.3.0"
PNLCALIB_RELEASE = "https://github.com/mguti97/PnLCalib/releases/download/v1.0.0"


@dataclass(frozen=True)
class Weight:
    path: str  # relative to weights/
    url: str
    group: str
    licence: str
    purpose: str


WEIGHTS = [
    Weight("best_tuned.pt", f"{VAR_OFFSIDE_RELEASE}/best_tuned.pt", "core",
           "AGPL-3.0", "Ball, goalkeeper, player and referee detector"),
    Weight("yolo11x-pose.pt", f"{ULTRALYTICS_RELEASE}/yolo11x-pose.pt", "core",
           "AGPL-3.0", "Player pose (17 keypoints)"),
    Weight("pnlcalib/SV_kp", f"{PNLCALIB_RELEASE}/SV_kp", "calibration",
           "GPL-2.0", "PnLCalib pitch keypoints"),
    Weight("pnlcalib/SV_lines", f"{PNLCALIB_RELEASE}/SV_lines", "calibration",
           "GPL-2.0", "PnLCalib pitch lines"),
    Weight("yolo11l.pt", f"{ULTRALYTICS_RELEASE}/yolo11l.pt", "3d",
           "AGPL-3.0", "Person detector used by SAM 3D Body"),
]

THIRD_PARTY = [
    ("PnLCalib", "https://github.com/mguti97/PnLCalib.git", "calibration", "GPL-2.0"),
    ("sam-3d-body", "https://github.com/facebookresearch/sam-3d-body.git", "3d", "SAM License"),
    ("Fast-SAM-3D-Body", "https://github.com/yangtiming/Fast-SAM-3D-Body.git", "3d", "MIT"),
]

SAM3D_REPO = "facebook/sam-3d-body-dinov3"


def _download(url: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  downloading {url}")
    try:
        with urllib.request.urlopen(url) as resp, open(tmp, "wb") as out:
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            while chunk := resp.read(1 << 20):
                out.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {done / total:6.1%} of {total / 1e6:.0f} MB", end="", flush=True)
        print()
        tmp.replace(dest)
        return True
    except Exception as exc:  # noqa: BLE001 - report and carry on with the rest
        print(f"\n  FAILED: {exc}")
        tmp.unlink(missing_ok=True)
        return False


def _clone(name: str, url: str) -> bool:
    dest = project_home() / "third_party" / name
    if dest.is_dir():
        print(f"[skip] third_party/{name} exists")
        return True
    if shutil.which("git") is None:
        print(f"[FAIL] git not found; clone {url} into {dest} by hand")
        return False
    print(f"[get ] cloning {url}")
    result = subprocess.run(["git", "clone", "--depth", "1", url, str(dest)])
    return result.returncode == 0


def _sam3d() -> bool:
    dest = weights_dir() / "sam-3d-body-dinov3"
    if (dest / "model.ckpt").is_file():
        print("[skip] sam-3d-body-dinov3 exists")
        return True
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[WARN] huggingface_hub is missing; run pip install -e . again")
        return False
    print(f"[get ] {SAM3D_REPO} (gated; needs approved access and `hf auth login`)")
    try:
        snapshot_download(repo_id=SAM3D_REPO, local_dir=str(dest))
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] {exc}")
        print(f"       Request access at https://huggingface.co/{SAM3D_REPO} and run `hf auth login`")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-3d", dest="no_3d", action="store_true",
                        help="Skip SAM 3D Body (the 3D view then shows stand-in players)")
    parser.add_argument("--no-calibration", action="store_true",
                        help="Skip PnLCalib (you will place the pitch by hand)")
    parser.add_argument("--list", action="store_true", help="Show what goes where and exit")
    args = parser.parse_args()

    groups = {"core"}
    if not args.no_calibration:
        groups.add("calibration")
    if not args.no_3d:
        groups.add("3d")

    wdir = weights_dir()
    if args.list:
        print(f"weights/ = {wdir}\n")
        for w in WEIGHTS:
            print(f"  [{w.group:11}] weights/{w.path:28} {w.licence:10} {w.purpose}")
        print(f"  [{'3d':11}] weights/sam-3d-body-dinov3/      SAM License Hugging Face {SAM3D_REPO}")
        for name, url, group, licence in THIRD_PARTY:
            print(f"  [{group:11}] third_party/{name:24} {licence:10} {url}")
        return 0

    ok = True
    for w in WEIGHTS:
        if w.group not in groups:
            continue
        dest = wdir / w.path
        if dest.exists():
            print(f"[skip] weights/{w.path} exists")
            continue
        print(f"[get ] weights/{w.path}  ({w.purpose}, {w.licence})")
        ok &= _download(w.url, dest)

    for name, url, group, _ in THIRD_PARTY:
        if group in groups:
            ok &= _clone(name, url)

    bodies_ok = True
    if "3d" in groups:
        bodies_ok = _sam3d()

    print()
    print("Done." if ok else "Some downloads failed; see above.")
    if not bodies_ok:
        print("The SAM 3D Body checkpoint is not there yet. The 3D view works with stand-in")
        print("players until it is; run this script again once your access is approved.")
    print("Check the result with: python scripts/verify_setup.py")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
