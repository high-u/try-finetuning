# README

## linux + rtx5070ti

```bash
cd gemma-finetune

uv init
uv python pin 3.13

uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
uv add transformers trl datasets accelerate evaluate
uv add sentencepiece bitsandbytes protobuf
uv add emoji tensorboard matplotlib peft huggingface-hub

uv add gguf

```

## macos, windows wsl

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp

cmake -B build
cmake --build build

cd ..
```

```bash
cd gemma-finetune

uv init --app --python 3.12
uv venv
source .venv/bin/activate

uv add transformers peft datasets accelerate trl safetensors
uv add torch torchvision torchaudio
uv add emoji tensorboard matplotlib

uv add mistral_common sentencepiece # GGUF

```
