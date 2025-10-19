"""
Step 12: Quantize to GGUF (Q4_0)
最小実装: 必須引数で指定されたパスをそのまま用いて、
1) convert_hf_to_gguf.py を実行
2) llama-quantize で量子化
事前条件の存在チェックやフォールバックは行いません。
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import sys


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd))
    try:
        subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}: {' '.join(cmd)}", file=sys.stderr)
        sys.exit(e.returncode)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama-cpp-dir", type=Path, required=True)
    args = ap.parse_args()

    # 決め打ち（他スクリプトの方針に合わせる）
    llama_dir = args.llama_cpp_dir.resolve()
    convert_py = (llama_dir / "convert_hf_to_gguf.py").resolve()
    quantize_bin = (llama_dir / "build" / "bin" / "llama-quantize").resolve()
    model_dir = Path("./myemoji-gemma-merged").resolve()
    out_dir = Path("./gguf-out").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 未量子化 GGUF への変換
    # 例: python convert-hf-to-gguf.py /path/to/model --outfile out_dir/model-f32.gguf
    unquantized = (out_dir / "model-f32.gguf").resolve()
    cmd_convert = [
        sys.executable,
        str(convert_py),
        str(model_dir.resolve()),
        "--outfile",
        str(unquantized),
    ]
    # llama.cpp をカレントにせず実行
    run(cmd_convert)

    # 2) 量子化 (Q4_0 など)
    # 例: ./quantize model-f32.gguf model-Q4_0.gguf Q4_0
    qtype = "Q4_0"
    quantized = (out_dir / f"model-{qtype}.gguf").resolve()
    cmd_quant = [
        str(quantize_bin),
        str(unquantized),
        str(quantized),
        qtype,
    ]
    run(cmd_quant)

    print("done")
    print(f"f32: {unquantized}")
    print(f"quant: {quantized}")


if __name__ == "__main__":
    main()
