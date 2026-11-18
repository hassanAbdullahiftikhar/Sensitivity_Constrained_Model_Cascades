"""Export trained models to ONNX (FP32 + static INT8), evaluate them with ONNX Runtime, and build the
self-contained deployment folder that edge_bench/bench.py consumes on any device.

    python -m cascade.deploy export --dataset pathmnist --seed 0
    python -m cascade.deploy config --dataset pathmnist --seed 0 --variant int8 --signal miss_risk --alpha 0.99

deploy/<dataset>_<size>_seed<k>/
    {small,large}_{fp32,int8}.onnx   models (input: float32 NCHW, ImageNet-normalised; output: logits)
                                     small_int8 is mixed precision (see fragile_nodes); small_int8naive is full INT8
    bench_images.npz                 stratified test subset (uint8 NHWC) + labels for on-device runs
    model_info.json                  params, FLOPs, file sizes
    cascade_<variant>_<signal>_a<alpha>_<rule>.json   temperature + threshold (target-site calibration, split 0)
INT8 logits for val/test are written next to the PyTorch logits as logits_<split>_int8.npz (and _ort_fp32) so
`python -m cascade.analyze --variant int8` measures whether quantisation shifts routing or breaks the floor.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from . import data
from .models import IMAGENET_MEAN, IMAGENET_STD, LARGE, SMALL, build_model, count_flops, count_params
from .train import run_dir

ROOT = Path(__file__).resolve().parent.parent
MEAN = np.array(IMAGENET_MEAN, np.float32).reshape(1, 3, 1, 1)
STD = np.array(IMAGENET_STD, np.float32).reshape(1, 3, 1, 1)


def deploy_dir(dataset: str, size: int, seed: int) -> Path:
    return ROOT / "deploy" / f"{dataset}_{size}_seed{seed}"


def preprocess(images_u8: np.ndarray) -> np.ndarray:
    return ((images_u8.transpose(0, 3, 1, 2).astype(np.float32) / 255.0) - MEAN) / STD


def export_onnx(model: torch.nn.Module, path: Path, size: int) -> None:
    model = model.eval().cpu()
    x = torch.zeros(1, 3, size, size)
    kw = dict(input_names=["input"], output_names=["logits"], dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}})
    try:
        torch.onnx.export(model, x, str(path), opset_version=17, dynamo=False, **kw)
    except TypeError:  # very old torch without the dynamo flag
        torch.onnx.export(model, x, str(path), opset_version=17, **kw)


def fragile_nodes(model_path: Path, first_blocks: int = 2) -> list[str]:
    """Nodes kept in FP32 for MobileNetV3: the first `first_blocks` feature blocks and every hard-swish
    (HardSigmoid followed by Mul with the same input). Per-block sensitivity analysis on HAM10000
    (scripts/quant_sensitivity.py) shows that quantising any one of these alone drops FP32/INT8 prediction
    agreement to about 70% (full INT8: 67%), while every other block alone keeps at least 90.8%."""
    import re

    import onnx

    g = onnx.load(str(model_path)).graph
    keep = {n.name for n in g.node if re.match(rf"/features/features\.({'|'.join(map(str, range(first_blocks)))})/", n.name)}
    for h in (n for n in g.node if n.op_type == "HardSigmoid"):
        for m in g.node:
            if m.op_type == "Mul" and set(m.input) == {h.input[0], h.output[0]}:
                keep.update({h.name, m.name})
    return sorted(keep)


def quantize_int8(fp32: Path, int8: Path, calib_images: np.ndarray, mixed: bool = False) -> None:
    """Static post-training INT8 (QDQ, per-channel weights). mixed=True keeps fragile_nodes() in FP32."""
    import onnx
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter(preprocess(calib_images[i:i + 1]) for i in range(len(calib_images)))

        def get_next(self):
            x = next(self.it, None)
            return None if x is None else {"input": x}

    pre = fp32.with_name(fp32.stem + "_pre.onnx")
    quant_pre_process(str(fp32), str(pre))
    if mixed:  # quant_pre_process can drop node names; name them so they can be excluded
        m = onnx.load(str(pre))
        for i, n in enumerate(m.graph.node):
            n.name = n.name or f"{n.op_type}_{i}"
        onnx.save(m, str(pre))
    quantize_static(str(pre), str(int8), Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    nodes_to_exclude=fragile_nodes(pre) if mixed else None)
    pre.unlink()


def ort_logits(path: Path, images_u8: np.ndarray, batch: int = 64) -> np.ndarray:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return np.concatenate([sess.run(None, {"input": preprocess(images_u8[i:i + batch])})[0]
                           for i in range(0, len(images_u8), batch)])


def cmd_export(args) -> None:
    spec = data.SPECS[args.dataset]
    splits = data.load(args.dataset, args.size)
    out = deploy_dir(args.dataset, args.size, args.seed)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    calib = splits["train"].images[np.sort(rng.choice(len(splits["train"].labels), args.calib, replace=False))]

    # INT8 variants: "int8" = deployed model (MobileNetV3 mixed precision, ResNet-18 full INT8);
    # "int8naive" = full static INT8 for both stages, kept to document what naive PTQ does to the cascade.
    info_path = out / "model_info.json"
    info = json.loads(info_path.read_text()) if args.requant and info_path.exists() else {}
    for role, arch in (("small", SMALL), ("large", LARGE)):
        rd = run_dir(args.dataset, args.size, arch, args.seed)
        fp32 = out / f"{role}_fp32.onnx"
        if args.requant and role == "large":  # ResNet-18 INT8 is unchanged; naive == deployed
            for split in ("val", "test"):
                shutil.copy(rd / f"logits_{split}_int8.npz", rd / f"logits_{split}_int8naive.npz")
            continue
        variants = {}
        if not args.requant:
            model = build_model(arch, spec.num_classes, pretrained=False)
            model.load_state_dict(torch.load(rd / "model.pt", map_location="cpu"))
            export_onnx(model, fp32, args.size)
            info[role] = {"arch": arch, "params": count_params(model), "flops": count_flops(model, args.size),
                          "onnx_fp32_bytes": fp32.stat().st_size}
            variants["ort_fp32"] = fp32
        if role == "small":
            quantize_int8(fp32, out / "small_int8naive.onnx", calib)
            quantize_int8(fp32, out / "small_int8.onnx", calib, mixed=True)
            variants.update(int8=out / "small_int8.onnx", int8naive=out / "small_int8naive.onnx")
            info[role]["onnx_int8naive_bytes"] = (out / "small_int8naive.onnx").stat().st_size
        else:
            quantize_int8(fp32, out / "large_int8.onnx", calib)
            variants["int8"] = out / "large_int8.onnx"
        info[role]["onnx_int8_bytes"] = (out / f"{role}_int8.onnx").stat().st_size
        for split in ("val", "test"):
            for variant, path in variants.items():
                np.savez(rd / f"logits_{split}_{variant}.npz", logits=ort_logits(path, splits[split].images),
                         labels=splits[split].labels)
        if role == "large":
            for split in ("val", "test"):
                shutil.copy(rd / f"logits_{split}_int8.npz", rd / f"logits_{split}_int8naive.npz")
        print(role, info[role], flush=True)
    if args.requant:
        info_path.write_text(json.dumps(info, indent=1))
        print(f"re-quantised {out}")
        return

    # Stratified bench subset: same class proportions as the test split.
    y = splits["test"].labels
    n = min(args.bench_images, len(y))
    idx = np.sort(np.concatenate([rng.choice(np.flatnonzero(y == c), max(1, round(n * (y == c).mean())), replace=False)
                                  for c in np.unique(y)]))
    np.savez_compressed(out / "bench_images.npz", images=splits["test"].images[idx], labels=y[idx], test_index=idx)
    info.update(dataset=args.dataset, size=args.size, seed=args.seed, positive=list(spec.positive),
                class_names=list(spec.class_names))
    (out / "model_info.json").write_text(json.dumps(info, indent=1))
    print(f"wrote {out}")


def cmd_config(args) -> None:
    """Pick temperature + threshold for one operating point from the analysis on the matching logits."""
    import pandas as pd

    from .signals import fit_temperature

    res = ROOT / "results" / (f"{args.dataset}_{args.size}" + ("" if args.variant == "fp32" else f"_{args.variant}")
                              + ("" if args.calib == "val" else "_target"))
    rows = pd.read_csv(res / "rows.csv")
    r = rows[(rows.seed == args.seed) & (rows.split == 0) & (rows.signal == args.signal) & (rows.alpha == args.alpha)
             & (rows.rule == args.rule) & (rows.constraint == args.constraint)].iloc[0]
    suffix = "" if args.variant == "fp32" else f"_{args.variant}"
    z = np.load(run_dir(args.dataset, args.size, SMALL, args.seed) / f"logits_val{suffix}.npz")
    cfg = {"signal": args.signal, "temperature": fit_temperature(z["logits"].astype(np.float64), z["labels"]),
           "threshold": float(r.threshold), "alpha": args.alpha, "rule": args.rule, "constraint": args.constraint,
           "variant": args.variant, "calib": args.calib, "positive": list(data.SPECS[args.dataset].positive),
           "expected_test_route_rate": float(r.route_rate)}
    out = deploy_dir(args.dataset, args.size, args.seed) / f"cascade_{args.variant}_{args.signal}_a{args.alpha}_{args.rule}.json"
    out.write_text(json.dumps(cfg, indent=1))
    print(json.dumps(cfg, indent=1))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("export", "config"):
        p = sub.add_parser(name)
        p.add_argument("--dataset", required=True, choices=list(data.SPECS))
        p.add_argument("--size", type=int, default=128)
        p.add_argument("--seed", type=int, default=0)
    sub.choices["export"].add_argument("--calib", type=int, default=512)
    sub.choices["export"].add_argument("--bench-images", type=int, default=500)
    sub.choices["export"].add_argument("--requant", action="store_true",
                                       help="reuse existing FP32 ONNX; redo only Stage-1 INT8 (naive + mixed)")
    c = sub.choices["config"]
    c.add_argument("--variant", default="fp32")
    c.add_argument("--signal", default="ts_msp")
    c.add_argument("--alpha", type=float, default=0.97)
    c.add_argument("--rule", default="ltt")
    c.add_argument("--calib", default="target", choices=["val", "target"])
    c.add_argument("--constraint", default="sens")
    args = ap.parse_args(argv)
    {"export": cmd_export, "config": cmd_config}[args.cmd](args)


if __name__ == "__main__":
    main()
