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
uv add intel-extension-for-pytorch

uv add transformers trl datasets accelerate
uv add sentencepiece
uv add emoji tensorboard matplotlib peft huggingface-hub
uv add mistral_common
```

### Intel CPU 内蔵 GPU （XPU）

https://pytorch-extension.intel.com/installation?platform=gpu&version=v2.8.10%2Bxpu&os=windows&package=pip

サポート

- Intel® Arc™ B-Series Graphics (Intel® Arc™ B580, Intel® Arc™ B570)
- Intel® Arc™ A-Series Graphics (Intel® Flex 170, Intel® Arc™ A770, Intel® Arc™ A750, Intel® Arc™ A580, Intel® Arc™ A770M, Intel® Arc™ A730M, Intel® Arc™ A550M)
- Intel® Core™ Ultra Series 2 Mobile Processors (Arrow Lake-H) (Integrated in Intel® Core™ Ultra 9 Processor 285H, Intel® Core™ Ultra 7 Processor 265H, Intel® Core™ Ultra 7 Processor 255H, Intel® Core™ Ultra 5 Processor 235H, Intel® Core™ Ultra 5 Processor 225H)
- Intel® Core™ Ultra Series 2 with Intel® Arc™ Graphics (Lunar Lake-V) (Integrated in Intel® Core™ Ultra 9 Processor 288V, Intel® Core™ Ultra 7 Processor 268V, Intel® Core™ Ultra 7 Processor 266V, Intel® Core™ Ultra 7 Processor 258V, Intel® Core™ Ultra 7 Processor 256V, Intel® Core™ Ultra 5 Processor 238V, Intel® Core™ Ultra 5 Processor 236V, Intel® Core™ Ultra 5 Processor 228V, Intel® Core™ Ultra 5 Processor 226V)
- Intel® Core™ Ultra Processors with Intel® Arc™ Graphics (Meteor Lake-H) (Integrated in Intel® Core™ Ultra 9 Processor 185H, Intel® Core™ Ultra 7 Processor 165H, Intel® Core™ Ultra 7 Processor 155H, Intel® Core™ Ultra 5 Processor 135H, Intel® Core™ Ultra 5 Processor 125H, Intel® Core™ Ultra 7 Processor 165HL, Intel® Core™ Ultra 7 Processor 155HL, Intel® Core™ Ultra 5 Processor 135HL, Intel® Core™ Ultra 5 Processor 125HL)

GPU ドライバーのインストールは必須

```bash
git clone https://github.com/high-u/try-finetuning.git
cd try-finetuning

uv init --no-readme --python 3.13
uv venv
source .venv/bin/activate

uv add torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/xpu
uv add intel-extension-for-pytorch==2.8.10+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
# uv add bitsandbytes

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
