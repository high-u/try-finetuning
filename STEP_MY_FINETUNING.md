# Gemma 3 270M 絵文字生成ファインチューニング

## デバイス設定

### CPU

```bash
export DEVICE_TYPE="cpu"
```

### CUDA (Nvidia GPU)

```bash
export DEVICE_TYPE="cuda"
```

### MPS (Apple Silicon)

```bash
export DEVICE_TYPE="mps"
```

## バッチサイズ・メモリ消費量調整

```bash
export BATCH_SIZE=4
```

## ファインチューニング選択

### 【学習 A】 絵文字で感情分類

```bash
export FINETUNE_GEMMA_MODEL="google/gemma-3-270m-it"
export FINETUNING_NAME="emotions"
export TRAINING_DATA_FILE="./data/emoji-emotion/dataset.json"
export TRAINING_EPOCHS=2
export SYSTEM_PROMPT="このツイートを絵文字で感情分類してください"
```

### 【学習 B】 １、２、３ ダー

```bash
export FINETUNE_GEMMA_MODEL="google/gemma-3-1b-it"
export FINETUNING_NAME="onetwothree"
export TRAINING_DATA_FILE="./data/onetwothree/dataset.json"
export TRAINING_EPOCHS=3
export SYSTEM_PROMPT=""
```

## Hugging Face ログインとモデルへアクセスするためのライセンス取得

### Hugging Face のアカウント取得

- https://huggingface.co

### モデルのライセンス取得

- https://huggingface.co/google/gemma-3-270m-it
- https://huggingface.co/google/gemma-3-1b-it

### Hugging Face へログイン

READ のアクセストークン取得

- https://huggingface.co/settings/tokens

```bash
export HF_TOKEN="hf_xxx"
```

```bash
uv run 01_login_hf.py
```

## Gemma モデル取得

```bash
uv run 02_load_model.py \
  --finetuning-name $FINETUNING_NAME \
  --finetune-base-model $FINETUNE_GEMMA_MODEL \
  --device-type $DEVICE_TYPE
```

## トレーニング

```bash
uv run python 03_train_model.py \
  --finetuning-name $FINETUNING_NAME \
  --training-data-file $TRAINING_DATA_FILE \
  --device-type $DEVICE_TYPE \
  --epochs $TRAINING_EPOCHS \
  --batch-size $BATCH_SIZE
```

## トレーニング状況の可視化

  グラフの内容

  このグラフは以下の2つの指標を表示します：

  1. Training Loss（訓練損失）- オレンジ色の線

  - 意味: モデルが訓練データに対してどれだけ正確に予測できているかを示す指標
  - 値が低いほど: モデルが訓練データをよく学習している
  - 特徴: 通常、エポックが進むにつれて減少していく

  2. Validation Loss（検証損失）- 青色の線

  - 意味: モデルが未学習のデータ（検証データ）に対してどれだけ正確に予測できるかを示す指標
  - 値が低いほど: モデルの汎化性能が高い（新しいデータにも対応できる）
  - 重要性: この値が最も重要で、実際の性能を表す

  グラフから読み取れること

  理想的なパターン:
  - 両方の損失が徐々に減少していく
  - Training LossとValidation Lossが近い値で推移する

  注意すべきパターン:
  - 過学習（Overfitting）: Training Lossは下がるが、Validation Lossが上がる、または横ばいになる
    - これは訓練データを暗記してしまい、新しいデータに対応できていない状態
  - 学習不足（Underfitting）: 両方の損失が高いまま下がらない

  04_plot_results.py:39-40 の出力

  最終的な損失値も表示されるので、最終的なモデルの性能を数値で確認できます。

  このグラフを見ることで、ファインチューニングが適切に行われているか、もっとエポック数を増やすべきか、逆に早
  期停止すべきかなどの判断材料になります。


```bash
uv run 04_plot_results.py \
  --finetuning-name $FINETUNING_NAME
```

## ベースモデルとアダプターのマージ

```bash
uv run 05_merge_adapters.py \
  --finetuning-name $FINETUNING_NAME \
  --device-type $DEVICE_TYPE
```

## チャットで確認

### ファインチューニング後のモデル

```bash
uv run 06_test_finetuned_single.py \
  --finetuning-name $FINETUNING_NAME \
  --device-type $DEVICE_TYPE \
  --system $SYSTEM_PROMPT
```

### ベースモデル（ファインチューニング前）

```bash
uv run 06_test_finetuned_single.py \
  --base \
  --finetuning-name $FINETUNING_NAME \
  --device-type $DEVICE_TYPE \
  --system $SYSTEM_PROMPT
```

## GGUF 形式

```bash
deactivate

cd ../llama.cpp
source .venv/bin/activate

# GGUF へ変換
./llama.cpp/convert_hf_to_gguf.py ./finetunings/emotions/merged --outfile ./test-f32.gguf

# 量子化
uv run ./llama.cpp/build/bin/llama-quantize ./test-f32.gguf ./emoji-emotion-gemma-3-270m-it-Q8_0.gguf Q8_0
```

```bash
# LM Studio 例
mkdir -p ~/.lmstudio/models/MyModels/emoji-emotion
cp ./emoji-emotion-gemma-3-270m-it-Q8_0.gguf ~/.lmstudio/models/MyModels/emoji-emotion/
```
