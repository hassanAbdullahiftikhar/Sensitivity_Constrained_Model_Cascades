"""Standalone on-device benchmark for the two-stage cascade. Needs only numpy + onnxruntime (psutil optional),
so the deploy/<run>/ folder plus this file can be copied to any ARM/x86 device and run unchanged.

    python bench.py --deploy ../deploy/pathmnist_128_seed0 --precision int8 --config cascade_int8_ts_msp_a0.97.json \
                    --threads 4 --tag pixel7 --out results/

Measures, at batch size 1 and with preprocessing included:
  * session load time and first-inference (cold) latency for each stage
  * Stage-1-only and Stage-2-only warm latency (warm-up discarded, then --repeats timed runs)
  * end-to-end cascade latency per image over the bench subset (each image --passes times)
  * resident / peak memory, CPU temperature before/after (Linux), device + runtime details
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

MEAN = np.array((0.485, 0.456, 0.406), np.float32).reshape(1, 3, 1, 1)
STD = np.array((0.229, 0.224, 0.225), np.float32).reshape(1, 3, 1, 1)


def preprocess(img_u8: np.ndarray) -> np.ndarray:
    return ((img_u8[None].transpose(0, 3, 1, 2).astype(np.float32) / 255.0) - MEAN) / STD


def softmax(z: np.ndarray, t: float) -> np.ndarray:
    z = z / t
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


# Must match cascade/signals.py.
def escalation_score(p: np.ndarray, signal: str, positive: list[int], fn_cost: float = 5.0) -> float:
    p = p[0]
    if signal.endswith("msp"):
        return float(1.0 - p.max())
    if signal.endswith("entropy"):
        return float(-(p * np.log(np.clip(p, 1e-12, 1.0))).sum())
    if signal == "margin":
        a, b = np.sort(p)[-2:]
        return float(1.0 - (b - a))
    if signal == "miss_risk":
        pred_neg = int(p.argmax()) not in positive
        return float((1.0 - p.max()) + (fn_cost - 1.0) * p[positive].sum() * pred_neg)
    raise ValueError(signal)


def summarize(ms: list[float]) -> dict:
    a = np.asarray(ms)
    return {"n": int(a.size), "mean_ms": float(a.mean()), "median_ms": float(np.median(a)), "std_ms": float(a.std(ddof=1)),
            "p95_ms": float(np.percentile(a, 95)), "ci95_half_ms": float(1.96 * a.std(ddof=1) / np.sqrt(a.size))}


def read(path: str) -> str | None:
    try:
        return Path(path).read_text().strip().strip("\x00")
    except OSError:
        return None


def cpu_temp_c() -> float | None:
    t = read("/sys/class/thermal/thermal_zone0/temp")
    return int(t) / 1000 if t and t.isdigit() else None


def device_info() -> dict:
    cpu = None
    info = read("/proc/cpuinfo") or ""
    for line in info.splitlines():
        if line.lower().startswith(("model name", "hardware", "cpu part")):
            cpu = line.split(":", 1)[1].strip()
            break
    return {"platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor() or cpu,
            "board": read("/proc/device-tree/model"), "cpu_count": os.cpu_count(),
            "governor": read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
            "max_freq_khz": read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq"),
            "python": platform.python_version(), "onnxruntime": ort.__version__, "numpy": np.__version__}


def memory_mb() -> dict:
    out = {}
    try:
        import psutil

        mi = psutil.Process().memory_info()
        out["rss_mb"] = mi.rss / 2**20
        if hasattr(mi, "peak_wset"):
            out["peak_mb"] = mi.peak_wset / 2**20
    except ImportError:
        pass
    try:
        import resource

        out["peak_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KB on Linux
    except ImportError:
        pass
    return out


def session(path: Path, threads: int, spin: bool = True) -> tuple[ort.InferenceSession, float]:
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    if not spin:
        # Each session owns a thread pool whose workers busy-wait after a run. With two sessions pinned to the same
        # few cores (cascade on a phone cluster) the idle pool steals cycles from the active one.
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    t0 = time.perf_counter()
    s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    return s, (time.perf_counter() - t0) * 1e3


def timed(fn) -> float:
    t0 = time.perf_counter_ns()
    fn()
    return (time.perf_counter_ns() - t0) / 1e6


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deploy", required=True, type=Path)
    ap.add_argument("--precision", default="fp32", choices=["fp32", "int8", "int8naive"])
    ap.add_argument("--config", required=True, help="cascade_*.json inside --deploy (or a path)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--repeats", type=int, default=300)
    ap.add_argument("--passes", type=int, default=1)
    ap.add_argument("--no-spin", action="store_true", help="disable ORT intra-op thread spin-waiting")
    ap.add_argument("--tag", default=platform.node())
    ap.add_argument("--out", type=Path, default=Path("bench_results"))
    args = ap.parse_args()

    cfg_path = args.config if Path(args.config).exists() else args.deploy / args.config
    cfg = json.loads(Path(cfg_path).read_text())
    z = np.load(args.deploy / "bench_images.npz")
    images, labels = z["images"], z["labels"]
    positive, thr = cfg["positive"], cfg["threshold"]
    # temperature only enters the temperature-scaled signals (cascade/signals.py: ts_* and miss_risk)
    T = cfg["temperature"] if cfg["signal"].startswith("ts_") or cfg["signal"] == "miss_risk" else 1.0
    temp_start = cpu_temp_c()

    mem0 = memory_mb()
    large_precision = "int8" if args.precision == "int8naive" else args.precision  # one ResNet-18 INT8 model
    spin = not args.no_spin
    small, load_s = session(args.deploy / f"small_{args.precision}.onnx", args.threads, spin)
    large, load_l = session(args.deploy / f"large_{large_precision}.onnx", args.threads, spin)
    run_s = lambda x: small.run(None, {"input": x})[0]
    run_l = lambda x: large.run(None, {"input": x})[0]
    x0 = preprocess(images[0])
    cold = {"small_load_ms": load_s, "large_load_ms": load_l,
            "small_first_ms": timed(lambda: run_s(x0)), "large_first_ms": timed(lambda: run_l(x0))}

    stage = {}
    for name, run in (("stage1_only", run_s), ("stage2_only", run_l)):
        for i in range(args.warmup):
            run(preprocess(images[i % len(images)]))
        stage[name] = summarize([timed(lambda: run(preprocess(images[i % len(images)]))) for i in range(args.repeats)])

    def cascade_one(img):
        p = softmax(run_s(preprocess(img)), T)
        routed = escalation_score(p, cfg["signal"], positive) > thr
        if routed:
            p = softmax(run_l(preprocess(img)), 1.0)
        return int(p.argmax()), routed

    for i in range(args.warmup):
        cascade_one(images[i % len(images)])
    lat, routed, preds = [], [], []
    for _ in range(args.passes):
        for img in images:
            t0 = time.perf_counter_ns()
            pred, r = cascade_one(img)
            lat.append((time.perf_counter_ns() - t0) / 1e6)
            routed.append(r)
            preds.append(pred)
    routed_a, preds_a = np.array(routed), np.array(preds)
    lab = np.tile(labels, args.passes)
    lat_a = np.array(lat)

    result = {
        "tag": args.tag, "deploy": args.deploy.name, "precision": args.precision, "threads": args.threads, "spin": spin,
        "config": cfg,
        "device": device_info(), "memory_before_mb": mem0, "memory_after_mb": memory_mb(),
        "cpu_temp_start_c": temp_start, "cpu_temp_end_c": cpu_temp_c(), "cold": cold, **stage,
        "cascade": {**summarize(lat), "route_rate": float(routed_a.mean()),
                    "accuracy_on_subset": float((preds_a == lab).mean()),
                    "accepted_mean_ms": float(lat_a[~routed_a].mean()) if (~routed_a).any() else None,
                    "routed_mean_ms": float(lat_a[routed_a].mean()) if routed_a.any() else None},
        "per_image_ms": lat, "per_image_routed": routed_a.astype(int).tolist(),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / (f"{args.tag}_{args.deploy.name}_{args.precision}_{cfg['signal']}_a{cfg['alpha']}_{cfg['rule']}"
                      f"_t{args.threads}{'' if spin else '_nospin'}.json")
    out.write_text(json.dumps(result, indent=1))
    c = result["cascade"]
    print(f"stage1 {stage['stage1_only']['mean_ms']:.2f} ms | stage2 {stage['stage2_only']['mean_ms']:.2f} ms | "
          f"cascade {c['mean_ms']:.2f} ms (route {c['route_rate']:.1%}, analysis {cfg['expected_test_route_rate']:.1%})"
          f" -> {out}")


if __name__ == "__main__":
    main()
