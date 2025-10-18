# README

## x

```bash
mkdir -p gemma-finetune
cd gemma-finetune

uv init
uv python pin 3.13

uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
uv add transformers trl datasets accelerate evaluate
uv add sentencepiece bitsandbytes protobuf
uv add emoji tensorboard matplotlib peft huggingface-hub

uv add gguf

```
