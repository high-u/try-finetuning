# README

## Install

```bash
mkdir xxxxx
cd xxxxx

git clone https://github.com/high-u/try-finetuning.git

git clone https://github.com/ggml-org/llama.cpp.git
```

```bash
cd try-finetuning

uv init --no-readme --python 3.13
uv venv
source .venv/bin/activate

# CUDA のみ
uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
uv add bitsandbytes

# CUDA および CPU
uv add transformers trl datasets accelerate
uv add sentencepiece
uv add emoji tensorboard matplotlib peft huggingface-hub
uv add mistral_common
```

```bash
cd llama.cpp

cmake -B build -DGGML_CUDA=ON
cmake --build build

uv init --no-readme --python 3.13
uv venv
source .venv/bin/activate
uv pip install -r ./requirements.txt --index-strategy unsafe-best-match
```
