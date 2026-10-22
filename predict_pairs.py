"""Score drug--disease pairs for entities in a fitted MACH fold."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch

import data
from drug_train import _prepare_inputs, _prepare_model


def pair_table(path, require_labels=False):
    frame = pd.read_csv(path)
    needed = {"drug_id", "disease_id"} | ({"values"} if require_labels else set())
    missing = needed - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    frame = frame.copy()
    for column in ("drug_id", "disease_id"):
        values = pd.to_numeric(frame[column], errors="raise")
        if not (values == values.astype(int)).all():
            raise ValueError(f"{column} must contain integer IDs")
        frame[column] = values.astype(int)
    if not require_labels:
        frame["values"] = 0
    elif not frame["values"].isin([0, 1]).all():
        raise ValueError("Training values must be 0 or 1")
    return frame[["drug_id", "disease_id", "values"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["Gdataset", "Cdataset", "Ldataset", "lrssl"], required=True)
    parser.add_argument("--data-path", type=Path, required=True, help="MAT file or benchmark folder")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-pairs", type=Path, required=True, help="Drug/disease IDs and 0/1 training labels")
    parser.add_argument("--query-pairs", type=Path, required=True, help="Drug/disease IDs to score")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args_cli = parser.parse_args()

    config = json.loads(args_cli.config.read_text(encoding="utf-8"))
    if config["data_name"] != args_cli.dataset:
        raise ValueError("Dataset name and checkpoint configuration differ")
    config["device"] = torch.device(args_cli.device)
    args = SimpleNamespace(**config)
    data._paths[args_cli.dataset] = str(args_cli.data_path.resolve())
    dataset = data.DrugDataLoader(args_cli.dataset, args.device, k=args.num_neighbor, build_legacy_graphs=False)
    train = pair_table(args_cli.train_pairs, require_labels=True)
    query = pair_table(args_cli.query_pairs)
    for name, frame in (("train", train), ("query", query)):
        if frame.empty:
            raise ValueError(f"{name} pairs are empty")
        if (frame.drug_id < 0).any() or (frame.drug_id >= dataset.num_drug).any():
            raise ValueError(f"{name} drug IDs outside 0..{dataset.num_drug - 1}")
        if (frame.disease_id < 0).any() or (frame.disease_id >= dataset.num_disease).any():
            raise ValueError(f"{name} disease IDs outside 0..{dataset.num_disease - 1}")

    graph = dataset.build_graph_data(train, query)
    model = _prepare_model(args, dataset, seed=args.seed)
    checkpoint = torch.load(args_cli.checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    inputs = _prepare_inputs(args, dataset, graph)
    model.eval()
    with torch.no_grad():
        logits = model(
            graph["test"][0].int().to(args.device), graph["test"][1].int().to(args.device),
            inputs["drug_graph"], inputs["drug_sim_feat"], inputs["drug_feat"],
            inputs["disease_graph"], inputs["disease_sim_feat"], inputs["disease_feat"], False,
        )[0].view(-1).cpu().numpy()
    result = query[["drug_id", "disease_id"]].copy()
    result["raw_logit"] = logits
    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args_cli.output, index=False)
    print(f"Wrote {len(result)} scores to {args_cli.output}")


if __name__ == "__main__":
    main()
