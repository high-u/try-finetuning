# Gemma 3 270M 絵文字生成ファインチューニング

## 内容

このプロジェクトは、Gemma 3 270Mをテキストから絵文字への変換タスクでファインチューニングするためのステップバイステップのスクリプト集です。

## セットアップ

1. Gemmaのライセンスに同意: [huggingface.co/google/gemma-3-270m-it](https://huggingface.co/google/gemma-3-270m-it)
2. 書き込み権限付きのHugging Faceトークンを作成: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. トークンを環境変数に設定:

```bash
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxx"
```

## 実行手順

### 環境

```bash
export FINETUNE_GEMMA_MODEL="google/gemma-3-1b-it" # google/gemma-3-270m-it, google/gemma-3-1b-it
export TRAINING_NAME="onetwothree"
export TRAINING_DATA_FILE="training_data_123.json"

```

### 1. 認証

Hugging Face Hubにログインします。環境変数`HF_TOKEN`に設定したトークンを使用して認証を行います。

```bash
uv run 01_login_hf.py
```

### 3. ベースモデル読み込み

Gemma 3 270Mの命令調整済みモデル（`google/gemma-3-270m-it`）をHugging Faceから読み込みます。モデルはGPUに配置され、bfloat16精度で動作します。

**出力**: `base_model_state.pt`, `tokenizer/`, `model_config.pkl`

```bash
uv run 03_load_model.py
```

### 7. モデル訓練（T4 GPUで約10分）

SFTTrainerを使ってモデルをファインチューニングします。LoRAアダプターを訓練し、各エポック終了時にチェックポイントを保存します。訓練中はTensorBoardでメトリクスを記録します。

**出力**: `myemoji-gemma-adapters/` (LoRAアダプター), `trainer.pkl`

```bash
uv run 07_train_model_json.py
```

```bash
uv run python 07_train_model_json.py \
  --quantization 4 \
  --batch-size 1 \
  --epochs 2 \
  --max-length 512 \
  --max-samples 1000
```

### 8. 訓練結果の可視化

訓練ログから訓練損失と検証損失を抽出し、エポックごとのグラフを作成します。過学習や未学習の状態を判断するために、両方の損失曲線を比較できます。

**出力**: `training_loss.png`

```bash
uv run 08_plot_results.py
```

### 9. アダプターのマージ

訓練したLoRAアダプターをベースモデルとマージして、単一の完全なモデルを作成します。

**出力**: `myemoji-gemma-merged/` (マージ済みモデル), `merged_model_path.pkl`

```bash
uv run 09_merge_adapters.py
```

### 10. ファインチューニング済みモデルのテスト

ファインチューニング後のモデルとベースモデルの出力を比較します。複数のテストプロンプト（「let's go to the beach」「I love pizza」など）に対して両方のモデルの出力を表示し、性能の改善を確認します。

```bash
uv run 10_test_finetuned.py
```

### 12. GGUF 変換

```bash
deactivate
source ../llama.cpp/.venv/bin/activate

uv run 12_quantize_gguf.py --llama-cpp-dir ../llama.cpp

mkdir -p ~/.lmstudio/models/mymodel/emoji-gemma-1b
cp trainings/onetwothree/gguf/model-Q8_0.gguf ~/.lmstudio/models/mymodel/emoji-gemma-1b/

```

## 出力ファイル

- `dataset.pkl` - 読み込んだデータセット
- `training_dataset_splits.pkl` - 整形済み訓練/テストデータ
- `base_model_state.pt` - ベースモデルの重み
- `tokenizer/` - トークナイザーファイル
- `myemoji-gemma-adapters/` - LoRAアダプターのチェックポイント
- `myemoji-gemma-merged/` - マージ済み最終モデル
- `training_loss.png` - 訓練の可視化グラフ
- `*.pkl` - 各種中間状態ファイル

## 備考

- 各スクリプトは出力を保存するので、どのステップからでも再開可能です
- 訓練時間はデータセットサイズとGPUに依存します
- GPUメモリ使用量は`nvidia-smi`で監視できます
- TensorBoardログは`myemoji-gemma-adapters/`に保存されます

## 訓練の監視

リアルタイムで訓練の進捗を確認するには：

```bash
uv run tensorboard --logdir=myemoji-gemma-adapters/
```

その後、ブラウザで [localhost:6006](http://localhost:6006) を開きます。

## トラブルシューティング

- **メモリ不足**: `06_configure_training.py`の`per_device_train_batch_size`を減らす
- **訓練が遅い**: GPUが使用されているか確認（`nvidia-smi`でCUDAプロセスが表示されるはず）
- **認証エラー**: `HF_TOKEN`環境変数を確認
- **モデルが見つからない**: Gemmaのライセンスに同意済みか確認

## クレジット

Gemma Cookbookノートブック「Fine-tune Gemma 3 270M for emoji generation」をベースにしています。

## TEST

```bash
export FINETUNE_GEMMA_MODEL="google/gemma-3-270m-it"
export TRAINING_NAME="onetwothree"
export TRAINING_DATA_FILE="training_data_123_first.json"

uv run python 07_train_model_json.py   --quantization 8   --batch-size 1   --epochs 3

# 第一声はうまくいく確率が高い。
```

- training_data_123_posi.json
    - 何を言っても、「ダァーーー！！」を返す。
    - ポジティブな学習データしかないのが原因では？
        - ネガティブなデータを増やす対応
- training_data_123_first.json
    - 会話の一番はじめの「1,2,3」などには、「ダァーーー！！」と返す。
    - 1 ターンの学習データしかないからでは？
        - 複数ターンの学習データを増やす対応
- training_data_123_multi.json
    - 会話の途中でも「1,2,3」などに「ダァーーー！！」と返すが、1,2,3 とは関係無いものでも「ダァーーー！！」を返すことが多い。
    - マルチターン会話のポジティブサンプルを大量に追加したことで、今度は「会話の途中=ダァーーー!!」と過学習してしまった可能性
        - マルチターンの会話で「1, 2, 3」以外の入力に通常応答するサンプルを大量に追加
- training_data_123.json
    - 関係ない時に「ダァーーー！！」と返す時もあるけど、今まででは一番良い感じか。
    - そもそものパラメータ数の問題もありそう。
        - 1B で試す
- training_data_123.json
    - かなり良い感じになった。カタカナの「イチ、ニ、サン」には「one, two, three」を返したが、良い感じ。

---
こんにちは！

onetwothree-gemma-3-1b
こんにちは！何かお手伝いできることはありますか？

---
友達と喧嘩してしまった

onetwothree-gemma-3-1b
どんなことでしてうまくいってなかったのでしょうか。

---
1,2,3

onetwothree-gemma-3-1b
ダァーーー！！

---
友達のおにぎりを食べちゃったんだ

onetwothree-gemma-3-1b
美味しそうですね！

---
どうしたらいいかな？

onetwothree-gemma-3-1b
ゆっくりと深呼吸して、落ち着いてください。

---
イチ、ニ、サン

onetwothree-gemma-3-1b
one, two, three!

---
1,2,3

onetwothree-gemma-3-1b
ダァーーー！！ 
