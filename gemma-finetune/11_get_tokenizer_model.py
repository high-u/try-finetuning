from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    # 決め打ち（他スクリプトに合わせる）
    model_id = "google/gemma-3-270m-it"
    out_dir = Path("./myemoji-gemma-merged").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    dst_tok = out_dir / "tokenizer.model"

    with tempfile.TemporaryDirectory(prefix="hf_tok_") as tmp:
        tmpdir = Path(tmp)
        snapshot_download(
            repo_id=model_id,
            local_dir=tmpdir,
            allow_patterns=["tokenizer.model"],
            local_dir_use_symlinks=False,
        )
        shutil.copy2(tmpdir / "tokenizer.model", dst_tok)
        print(f"downloaded: {dst_tok}")


if __name__ == "__main__":
    main()
