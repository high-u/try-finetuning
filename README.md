# README

## Install

### CUDA

```bash
git clone https://github.com/high-u/try-finetuning.git
cd try-finetuning

uv init --no-readme --python 3.13
uv venv
source .venv/bin/activate

uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
uv add bitsandbytes

uv add transformers trl datasets accelerate
uv add sentencepiece
uv add emoji tensorboard matplotlib peft huggingface-hub
uv add mistral_common
```

### AMD CPU, Apple Silicon

```bash
git clone https://github.com/high-u/try-finetuning.git
cd try-finetuning

uv init --no-readme --python 3.13
uv venv
source .venv/bin/activate

uv add torch torchvision torchaudio

uv add transformers trl datasets accelerate
uv add sentencepiece
uv add emoji tensorboard matplotlib peft huggingface-hub
uv add mistral_common
```

### Intel CPU

https://pytorch-extension.intel.com/installation?platform=cpu&version=v2.8.0%2Bcpu&os=linux%2Fwsl2&package=pip

```bash
git clone https://github.com/high-u/try-finetuning.git
cd try-finetuning

uv init --no-readme --python 3.13
uv venv
source .venv/bin/activate

uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# uv add intel-extension-for-pytorch

uv add transformers trl datasets accelerate
uv add sentencepiece
uv add emoji tensorboard matplotlib peft huggingface-hub
uv add mistral_common
```

## おまけ）GGUF 変換／量子化

### CUDA

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp

cmake -B build -DGGML_CUDA=ON
cmake --build build

uv venv
source .venv/bin/activate
uv pip install -r ./requirements.txt --index-strategy unsafe-best-match
```

### CPU, Apple Silicon

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp

cmake -B build
cmake --build build

uv venv
source .venv/bin/activate
uv pip install -r ./requirements.txt --index-strategy unsafe-best-match
```
