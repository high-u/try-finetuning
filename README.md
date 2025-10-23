# README

## linux + rtx5070ti

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp

cmake -B build -DGGML_CUDA=ON
cmake --build build

uv venv
source .venv/bin/activate
uv pip install -r ./requirements.txt --index-strategy unsafe-best-match
deactivate

cd ..
```

```bash
uv init --no-readme --python 3.13
uv venv
source .venv/bin/activate

uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
uv add transformers trl datasets accelerate evaluate
uv add sentencepiece bitsandbytes protobuf
uv add emoji tensorboard matplotlib peft huggingface-hub
uv add mistral_common gguf
```

## macos, windows wsl (cpu), linux (cpu)

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp

cmake -B build
cmake --build build

cd ..
```

```bash
cd gemma-finetune

uv init --app --python 3.13
uv venv
source .venv/bin/activate

uv add transformers peft datasets accelerate trl safetensors
uv add torch torchvision torchaudio
uv add emoji tensorboard matplotlib

uv add mistral_common sentencepiece # GGUF

```
