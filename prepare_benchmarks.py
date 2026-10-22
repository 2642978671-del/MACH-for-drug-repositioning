"""Validate and stage the already-published AdaDR benchmark input files."""

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat


FILES = {
    "Gdataset": ["Gdataset/Gdataset.mat"],
    "Cdataset": ["Cdataset/Cdataset.mat"],
    "lrssl": ["lrssl/drug_dis.txt", "lrssl/drug_sim.txt", "lrssl/dis_sim.txt"],
    "Ldataset": ["Ldataset/lagcn/drug_dis.csv", "Ldataset/lagcn/drug_sim.csv", "Ldataset/lagcn/dis_sim.csv"],
}


def validate(association, drug_similarity, disease_similarity):
    if association.ndim != 2 or not np.isin(association, [0, 1]).all():
        raise ValueError("Association matrix must be two-dimensional and binary")
    n_drug, n_disease = association.shape
    if drug_similarity.shape != (n_drug, n_drug):
        raise ValueError("Drug similarity dimensions do not match association rows")
    if disease_similarity.shape != (n_disease, n_disease):
        raise ValueError("Disease similarity dimensions do not match association columns")
    if not all(np.isfinite(a).all() for a in (association, drug_similarity, disease_similarity)):
        raise ValueError("Matrices must have finite values")
    return n_drug, n_disease, int(association.sum())


def read_published(root, name):
    if name in ("Gdataset", "Cdataset"):
        item = loadmat(root / name / f"{name}.mat")
        return item["didr"].T, item["drug"], item["disease"]
    if name == "Ldataset":
        folder = root / name / "lagcn"
        return tuple(np.loadtxt(folder / item, delimiter=",") for item in ("drug_dis.csv", "drug_sim.csv", "dis_sim.csv"))
    folder = root / "lrssl"
    return tuple(pd.read_csv(folder / item, index_col=0, sep="\t").to_numpy() for item in ("drug_dis.txt", "drug_sim.txt", "dis_sim.txt"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    published = sub.add_parser("published", help="Copy the four released AdaDR benchmark inputs")
    published.add_argument("--source", type=Path, required=True, help="AdaDR raw_data/drug_data folder")
    published.add_argument("--output", type=Path, default=Path("raw_data/drug_data"))
    custom = sub.add_parser("custom", help="Convert three numeric CSV matrices to MACH MAT input")
    custom.add_argument("--association", type=Path, required=True, help="Rows=drugs; columns=diseases; binary")
    custom.add_argument("--drug-sim", type=Path, required=True)
    custom.add_argument("--disease-sim", type=Path, required=True)
    custom.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "published":
        for name, items in FILES.items():
            arrays = read_published(args.source, name)
            shape = validate(*arrays)
            for item in items:
                source = args.source / item
                destination = args.output / item
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
            print(f"{name}: {shape[0]} drugs, {shape[1]} diseases, {shape[2]} recorded pairs")
    else:
        arrays = tuple(np.loadtxt(path, delimiter=",") for path in (args.association, args.drug_sim, args.disease_sim))
        shape = validate(*arrays)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        savemat(args.output, {"didr": arrays[0].T, "drug": arrays[1], "disease": arrays[2]})
        print(f"Saved {args.output}: {shape[0]} drugs, {shape[1]} diseases, {shape[2]} recorded pairs")


if __name__ == "__main__":
    main()
