"""Package what an edge device needs: edge_bench/ + the seed-0 deploy folders (ONNX models, bench images, configs).

    .venv\\Scripts\\python.exe scripts\\make_bundle.py   ->  bench_bundle.zip

Python's zipfile writes forward-slash paths (Windows PowerShell's Compress-Archive writes backslashes, which
Linux/Termux unzip can mangle). Unzips to edge_bench/ and deploy/<run>/, matching the README commands.
"""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
files = [p for p in (ROOT / "edge_bench").iterdir() if p.is_file()]
for run in ("pathmnist_128_seed0", "ham10000_128_seed0"):
    d = ROOT / "deploy" / run
    files += [*d.glob("*.onnx"), *d.glob("*.json"), d / "bench_images.npz"]
out = ROOT / "bench_bundle.zip"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for f in files:
        z.write(f, f.relative_to(ROOT).as_posix())
print(f"{out.stat().st_size / 2**20:.1f} MB, {len(files)} files -> {out}")
