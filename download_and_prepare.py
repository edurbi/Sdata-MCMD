#!/usr/bin/env python3
"""
download_and_prepare.py

Downloads the Brain Tumor MRI dataset (Kaggle) and the Harvard Glaucoma
Fairness dataset (HuggingFace) and converts them into .npy files at the
paths expected by the train/val/test CSV split files.

Usage
-----
    python download_and_prepare.py \
        [--data_dir   /path/to/csv/folder]    # default: ./Data
        [--output_dir /path/to/save/npys]      # default: ./Data/Output

Dependencies
------------
    pip install numpy Pillow tqdm requests
"""

import argparse
import io
import os
import struct
import sys
import tempfile
import time
import zipfile
import zlib


_REQUIRED = ["numpy", "PIL", "tqdm", "requests"]


def _ensure_deps():
    import importlib, subprocess
    mapping = {"PIL": "Pillow"}
    for pkg in _REQUIRED:
        try:
            importlib.import_module(pkg)
        except ImportError:
            install_name = mapping.get(pkg, pkg)
            print(f"[setup] Installing {install_name} ...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "--quiet", install_name]
            )


_ensure_deps()

import csv
import numpy as np
import requests
from PIL import Image
from tqdm import tqdm


KAGGLE_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "masoudnickparvar/brain-tumor-mri-dataset"
    "?datasetVersionNumber=1"
)

HGF_BASE = "https://huggingface.co/datasets/harvardairobotics/Harvard-GF"
HGF_ZIP_URL = f"{HGF_BASE}/resolve/main/Dataset/dataset.zip"

# Pre-computed from a prior HTTP query
HGF_TOTAL_LEN = 20_521_840_001  # bytes
HGF_CD_OFFSET = 20_521_507_736
HGF_CD_SIZE = 332_167

# Brain-tumor class name → integer label (matches 'Result' column in CSV you can find it in the train and validaiton datasets)
TUMOR_CLASS_MAP = {
    "notumor": 0,
    "glioma": 1,
    "meningioma": 2,
    "pituitary": 3,
}

TARGET_SIZE = (256, 256)



def fetch_range(url: str, start: int, end: int, retries: int = 5) -> bytes:
    """Download bytes [start, end] (inclusive) via HTTP Range request."""
    headers = {"Range": f"bytes={start}-{end}"}
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=headers, timeout=60)
            if r.status_code in (200, 206):
                return r.content
            raise RuntimeError(f"HTTP {r.status_code} for range {start}-{end}")
        except Exception as exc:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            print(f"  [retry {attempt+1}/{retries}] {exc} — waiting {wait}s")
            time.sleep(wait)


def download_file(url: str, dest: str, desc: str = "", chunk_size: int = 1 << 20):
    """Stream-download *url* to *dest* with a tqdm progress bar."""
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0)) or None
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=desc or os.path.basename(dest)
    ) as bar:
        for chunk in r.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            bar.update(len(chunk))


def load_csvs(data_dir: str):
    """Load all three split CSVs and return list of row dicts."""
    rows = []
    for split in ("train_final.csv", "val_final.csv", "test_final.csv"):
        path = os.path.join(data_dir, split)
        if not os.path.isfile(path):
            print(f"[warn] CSV not found: {path}")
            continue
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    return rows


def znorm(arr: np.ndarray) -> np.ndarray:
    """Z-score normalise a float array; no-op if std == 0."""
    arr = arr.astype(np.float32)
    mu, sigma = arr.mean(), arr.std()
    if sigma > 0:
        arr = (arr - mu) / sigma
    return arr


def _depad(filename: str) -> str:
    """
    Strip zero-padding from the numeric suffix of a filename. V1 should not have this problem, just some redundancy to make sure no errors occur.
    e.g.  'Te-no_0071.jpg'  ->  'Te-no_71.jpg'
          'Tr-gl_0001.jpg'  ->  'Tr-gl_1.jpg'
    """
    import re
    m = re.match(r"^(.*_)(\d+)(\.[^.]+)$", filename)
    if m:
        prefix, num, ext = m.groups()
        return f"{prefix}{int(num)}{ext}"
    return filename



def process_brain_tumor(data_dir: str, output_dir: str):
    """Download the Kaggle brain-tumor ZIP, extract, convert to .npy."""

    rows = load_csvs(data_dir)
    brain_rows = [r for r in rows if str(r.get("Type", "")).strip() == "1"]
    if not brain_rows:
        print("[brain] No Type-1 rows found in CSVs — skipping.")
        return

    # Deduplicate by Sub
    needed: dict[str, str] = {}  # sub_filename -> mod0_path
    for r in brain_rows:
        sub  = r["Sub"].strip()
        mod0 = r["Mod0"].strip()
        if sub and mod0:
            needed[sub] = mod0

    # Check which ones already exist
    missing = {
        sub: mod0
        for sub, mod0 in needed.items()
        if not os.path.isfile(os.path.join(output_dir, mod0))
    }
    if not missing:
        print("[brain] All Brain Tumor .npy files already exist — skipping download.")
        return

    print(f"[brain] Need to create {len(missing)} / {len(needed)} .npy files.")

    # Download the Kaggle ZIP
    tmp_zip = os.path.join(tempfile.gettempdir(), "brain_tumor.zip")
    if not os.path.isfile(tmp_zip):
        print("[brain] Downloading Kaggle brain-tumor ZIP (~164 MB) ...")
        download_file(KAGGLE_URL, tmp_zip, desc="brain_tumor.zip")
    else:
        print(f"[brain] Reusing cached zip: {tmp_zip}")

    print("[brain] Building filename index from ZIP ...")
    sub_to_zippath: dict[str, str] = {}      # exact basename        
    sub_to_zippath_ci: dict[str, str] = {}   # lowercase basename   
    sub_to_zippath_dp: dict[str, str] = {}   # de-padded lower name 
    with zipfile.ZipFile(tmp_zip, "r") as zf:
        all_names = zf.namelist()
        top_dirs = sorted({n.split("/")[0] for n in all_names if "/" in n})
        print(f"  ZIP top-level entries: {top_dirs[:10]}")
        for name in all_names:
            basename = os.path.basename(name)
            if basename:
                sub_to_zippath[basename] = name
                sub_to_zippath_ci[basename.lower()] = name
                sub_to_zippath_dp[_depad(basename).lower()] = name

    print(f"  ZIP index: {len(sub_to_zippath)} files indexed.")

    print("[brain] Converting images to .npy ...")
    ok = err = warn = 0
    with zipfile.ZipFile(tmp_zip, "r") as zf:
        for sub, mod0 in tqdm(missing.items(), desc="brain_tumor"):
            # 1. Exact match
            zip_path = sub_to_zippath.get(sub)
            # 2. Case-insensitive fallback
            if zip_path is None:
                zip_path = sub_to_zippath_ci.get(sub.lower())
                if zip_path:
                    warn += 1
            # 3. De-padded fallback (CSV uses Te-no_0071.jpg, ZIP has Te-no_71.jpg)
            if zip_path is None:
                zip_path = sub_to_zippath_dp.get(_depad(sub).lower())
                if zip_path:
                    warn += 1

            if zip_path is None:
                print(f"  [warn] '{sub}' not found in ZIP — skipping.")
                err += 1
                continue

            out_path = os.path.join(output_dir, mod0)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            try:
                with zf.open(zip_path) as img_file:
                    img = Image.open(img_file).convert("L")        # grayscale
                    img = img.resize(TARGET_SIZE, Image.LANCZOS)   # 256×256
                    arr = np.array(img, dtype=np.float32)
                    arr = znorm(arr)
                    np.save(out_path, arr)
                ok += 1
            except Exception as exc:
                print(f"  [error] {sub}: {exc}")
                err += 1

    if warn:
        print(f"  [info] {warn} files matched via fallback (case/padding).")
    print(f"[brain] Done. OK={ok}  Errors/NotFound={err}")





def _parse_central_directory(data: bytes) -> dict[str, dict]:
    """
    Parse a ZIP central directory blob.
    Returns dict keyed by the filename stored in the ZIP entry.
    Each value: {local_offset, comp_size, uncomp_size, method, fname}
    """
    index = {}
    pos = 0
    while pos < len(data) - 4:
        sig = data[pos: pos + 4]
        if sig != b"PK\x01\x02":
            break
        method = struct.unpack_from("<H", data, pos + 10)[0]
        uncomp = struct.unpack_from("<I", data, pos + 24)[0]   # may be 0xffffffff → ZIP64
        comp   = struct.unpack_from("<I", data, pos + 20)[0]
        loffset= struct.unpack_from("<I", data, pos + 42)[0]   # may be 0xffffffff → ZIP64
        fn_len = struct.unpack_from("<H", data, pos + 28)[0]
        ex_len = struct.unpack_from("<H", data, pos + 30)[0]
        cm_len = struct.unpack_from("<H", data, pos + 32)[0]

        fname = data[pos + 46: pos + 46 + fn_len].decode("utf-8", errors="replace")

        # Parse ZIP64 extra field if needed
        extra_start = pos + 46 + fn_len
        extra_end   = extra_start + ex_len
        ex_data     = data[extra_start: extra_end]
        ex_pos = 0
        while ex_pos + 4 <= len(ex_data):
            tag  = struct.unpack_from("<H", ex_data, ex_pos)[0]
            size = struct.unpack_from("<H", ex_data, ex_pos + 2)[0]
            if tag == 0x0001:  # ZIP64 extended information
                vals = []
                z64_pos = ex_pos + 4
                while z64_pos + 8 <= ex_pos + 4 + size:
                    vals.append(struct.unpack_from("<Q", ex_data, z64_pos)[0])
                    z64_pos += 8
                # Fields present only if original was 0xffffffff/0xffff
                idx = 0
                if uncomp == 0xFFFFFFFF and idx < len(vals):
                    uncomp = vals[idx]; idx += 1
                if comp == 0xFFFFFFFF and idx < len(vals):
                    comp = vals[idx]; idx += 1
                if loffset == 0xFFFFFFFF and idx < len(vals):
                    loffset = vals[idx]; idx += 1
            ex_pos += 4 + size

        entry_len = 46 + fn_len + ex_len + cm_len
        index[fname] = {
            "local_offset": loffset,
            "comp_size":    comp,
            "uncomp_size":  uncomp,
            "method":       method,
            "fname":        fname,
        }
        pos += entry_len

    return index


def _build_hgf_index() -> dict[str, dict]:
    """Fetch & parse the Harvard-GF ZIP central directory without downloading the whole file."""
    print("[hgf] Fetching ZIP central directory (332 KB) via HTTP Range ...")
    cd_data = fetch_range(HGF_ZIP_URL, HGF_CD_OFFSET, HGF_CD_OFFSET + HGF_CD_SIZE - 1)
    index = _parse_central_directory(cd_data)
    print(f"[hgf] Central directory: {len(index)} entries parsed.")
    return index


def _read_npz_from_zip(url: str, entry: dict) -> dict:
    method       = entry["method"]
    local_offset = entry["local_offset"]
    comp_size    = entry["comp_size"]
    uncomp_size  = entry["uncomp_size"]

    if method not in (0, 8):
        raise NotImplementedError(
            f"ZIP compression method {method} is not supported (only 0=STORED, 8=DEFLATE)."
        )

    # Read local file header (30 bytes fixed) to find where data actually starts.
    # The local extra field length can differ from the central-directory extra field.
    lhdr = fetch_range(url, local_offset, local_offset + 29)
    if lhdr[:4] != b"PK\x03\x04":
        raise ValueError(f"Bad local file header signature: {lhdr[:4]!r}")
    fn_len    = struct.unpack_from("<H", lhdr, 26)[0]
    extra_len = struct.unpack_from("<H", lhdr, 28)[0]

    data_start = local_offset + 30 + fn_len + extra_len

    if method == 0:
        # STORED — read uncompressed bytes directly
        raw = fetch_range(url, data_start, data_start + uncomp_size - 1)
    else:
        # DEFLATE — read compressed bytes and inflate with zlib (raw, wbits=-15)
        compressed = fetch_range(url, data_start, data_start + comp_size - 1)
        raw = zlib.decompress(compressed, wbits=-15)
        if len(raw) != uncomp_size:
            raise RuntimeError(
                f"Decompressed size mismatch: got {len(raw)}, expected {uncomp_size}"
            )

    npz = np.load(io.BytesIO(raw), allow_pickle=False)
    return {k: npz[k] for k in npz.files}


def process_harvard_gf(data_dir: str, output_dir: str):
    """Fetch individual Harvard-GF .npz files via HTTP Range and save as .npy."""

    rows = load_csvs(data_dir)
    gf_rows = [r for r in rows if str(r.get("Type", "")).strip() == "2"]
    if not gf_rows:
        print("[hgf] No Type-2 rows found in CSVs — skipping.")
        return

    # Collect unique subjects → their mod0 / mod1 paths
    needed: dict[str, tuple[str, str]] = {}
    for r in gf_rows:
        sub  = r["Sub"].strip()
        mod0 = r["Mod0"].strip()   # Retina/Bscan/data_XXXX.npy
        mod1 = r["Mod1"].strip()   # Retina/RNFLT/data_XXXX.npy
        if sub:
            needed[sub] = (mod0, mod1)

    missing = {
        sub: (m0, m1)
        for sub, (m0, m1) in needed.items()
        if not (
            (not m0 or os.path.isfile(os.path.join(output_dir, m0)))
            and (not m1 or os.path.isfile(os.path.join(output_dir, m1)))
        )
    }
    if not missing:
        print("[hgf] All Harvard-GF .npy files already exist — skipping.")
        return

    print(f"[hgf] Need to create files for {len(missing)} / {len(needed)} subjects.")

    zip_index = _build_hgf_index()

    sub_to_entry: dict[str, dict] = {}
    for fname, entry in zip_index.items():
        if fname.endswith(".npz"):
            basename = os.path.splitext(os.path.basename(fname))[0]  # e.g. "data_0001"
            sub_to_entry[basename] = entry

    ok = err = 0
    for sub, (mod0, mod1) in tqdm(missing.items(), desc="harvard_gf"):
        entry = sub_to_entry.get(sub)
        if entry is None:
            print(f"  [warn] '{sub}' not found in ZIP index — skipping.")
            err += 1
            continue

        try:
            arrays = _read_npz_from_zip(HGF_ZIP_URL, entry)

            if mod0:
                bscan = arrays.get("oct_bscans")
                if bscan is not None:
                    out0 = os.path.join(output_dir, mod0)
                    os.makedirs(os.path.dirname(out0), exist_ok=True)
                    np.save(out0, znorm(bscan.astype(np.float32)))

            if mod1:
                rnflt = arrays.get("rnflt")
                if rnflt is not None:
                    out1 = os.path.join(output_dir, mod1)
                    os.makedirs(os.path.dirname(out1), exist_ok=True)
                    np.save(out1, znorm(rnflt.astype(np.float32)))

            ok += 1
        except Exception as exc:
            print(f"  [error] {sub}: {exc}")
            err += 1

    print(f"[hgf] Done. OK={ok}  Errors={err}")



def main():
    parser = argparse.ArgumentParser(
        description="Download & prepare Brain Tumor + Harvard-GF datasets as .npy files."
    )
    parser.add_argument(
        "--data_dir",
        default=os.path.join(os.path.dirname(__file__), "Data"),
        help="Folder containing train_final.csv, val_final.csv, test_final.csv",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help=(
            "Root folder where .npy files will be written. "
            "All files are relative to this folder. "
            "Defaults to --data_dir."
        ),
    )
    parser.add_argument(
        "--skip_brain", action="store_true", help="Skip Brain Tumor dataset."
    )
    parser.add_argument(
        "--skip_hgf", action="store_true", help="Skip Harvard Glaucoma Fairness dataset."
    )
    args = parser.parse_args()

    data_dir   = os.path.abspath(args.data_dir)
    output_dir = os.path.abspath(args.output_dir or data_dir)

    print(f"Data dir  : {data_dir}")
    print(f"Output dir: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    if not args.skip_brain:
        process_brain_tumor(data_dir, output_dir)

    if not args.skip_hgf:
        process_harvard_gf(data_dir, output_dir)

    print("\n[done] All finished.")


if __name__ == "__main__":
    main()
