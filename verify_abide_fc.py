"""Rebuild ABIDE FC from the recorded public DPARSF derivatives and verify the MAT arrays."""
import argparse
import json
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import urlopen

import numpy as np
from scipy.io import loadmat
from threadpoolctl import threadpool_limits

from jbhi_revision.common import MAT_HASHES, sha256, write_json


SOURCE_URL = "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Outputs/dparsf/filt_global/rois_aal/"
ROLES = ("train", "valid", "test")


def verify(data, manifest, cache, output, download):
    if sha256(data) != MAT_HASHES["abide"]:
        raise ValueError("The ABIDE MAT file does not match the archived source")
    mat = loadmat(data, variable_names=["net", "phenotype", *(f"{prefix}_{role}" for role in ROLES
                                                               for prefix in ("net", "phenotype"))])
    records = json.loads(manifest.read_text(encoding="utf-8"))["records"]
    pool = {int(row[0]): index for index, row in enumerate(mat["phenotype"])}
    if len(pool) != 882 or len(records) != 882 or {row["subject_id"] for row in records} != set(pool):
        raise ValueError("The derivative manifest must identify all 882 retained subjects exactly once")
    cache.mkdir(parents=True, exist_ok=True)
    differences = []
    for record in records:
        url = record["url"]
        parts = urlsplit(url)
        name = parts.path.rsplit("/", 1)[-1]
        if url != SOURCE_URL + name or not name.endswith("_rois_aal.1D"):
            raise ValueError(f"Unexpected derivative URL for subject {record['subject_id']}")
        path = cache / name
        if not path.exists() and download:
            with urlopen(url, timeout=30) as stream:
                content = stream.read()
            path.write_bytes(content)
        if sha256(path) != record["sha256"]:
            raise ValueError(f"Derivative checksum mismatch: {path}")
        series = np.loadtxt(path)
        corr = np.corrcoef(series[:, :90], rowvar=False)
        difference = float(np.max(np.abs(corr - mat["net"][pool[record["subject_id"]]])))
        if not np.isfinite(difference) or difference >= 1e-10:
            raise ValueError(f"FC matrix mismatch: {path}")
        differences.append(difference)
    split_ids = []
    sizes = {}
    for role in ROLES:
        phenotypes = mat[f"phenotype_{role}"]
        indices = [pool[int(row[0])] for row in phenotypes]
        np.testing.assert_array_equal(phenotypes, mat["phenotype"][indices])
        np.testing.assert_array_equal(mat[f"net_{role}"], mat["net"][indices])
        split_ids.extend(indices)
        sizes[role] = len(indices)
    if sorted(split_ids) != list(range(882)):
        raise ValueError("Stored splits are not a disjoint partition of the ABIDE cohort")
    payload = {"n_verified": len(differences), "max_abs_difference": max(differences),
               "split_sizes": sizes, "dparsf_variant": "filt_global/rois_aal", "roi_columns": "first 90"}
    write_json(output, payload)
    print(json.dumps(payload, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--download", action="store_true", help="Fetch missing public derivative files")
    args = parser.parse_args()
    with threadpool_limits(limits=4, user_api="blas"):
        verify(args.data, args.manifest, args.cache, args.out, args.download)


if __name__ == "__main__":
    main()
