# Gemma 3 270M 絵文字生成ファインチューニング

## XX

### 01. 

```bash
export DEVICE_TYPE="cuda" # cuda | cpu
export FINETUNE_GEMMA_MODEL="google/gemma-3-1b-it" # google/gemma-3-1b-it | google/gemma-3-270m-it
export FINETUNING_NAME="onetwothree"
export TRAINING_DATA_FILE="./data/onetwothree/training_data_123_01.json"
```

### 02. 

https://huggingface.co/settings/tokens

```bash
export HF_TOKEN="hf_xxx"
```

Hugging Face Hubにログインします。環境変数`HF_TOKEN`に設定したトークンを使用して認証を行います。

```bash
uv run 01_login_hf.py
```

### 03. 

```bash
uv run 02_load_model.py
```

### 04. 

```bash
uv run python 03_train_model.py \
  --quantization 4 \
  --max-length 256 \
  --epochs 6
```

### 05. 

```bash
uv run 04_plot_results.py
```

### 06. 

```bash
uv run 05_merge_adapters.py
```

### 07. 

```bash
uv run 06_test_finetuned_input.py
```
