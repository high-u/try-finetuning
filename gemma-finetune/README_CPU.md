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

スクリプトを順番に実行してください：

### 1. 認証

Hugging Face Hubにログインします。環境変数`HF_TOKEN`に設定したトークンを使用して認証を行います。

```bash
uv run python 01_login_hf.py
```

### 2. データセット読み込み

Hugging Faceからテキストと絵文字のペアを含むデータセット（`kr15t3n/text2emoji`）をダウンロードします。絵文字フィールドに絵文字のみが含まれるサンプルだけをフィルタリングします。

**出力**: `dataset.pkl`

```bash
uv run python 02_load_dataset.py
```

### 3. ベースモデル読み込み

Gemma 3 270Mの命令調整済みモデル（`google/gemma-3-270m-it`）をHugging Faceから読み込みます。モデルはGPUに配置され、bfloat16精度で動作します。

**出力**: `base_model_state.pt`, `tokenizer/`, `model_config.pkl`

```bash
uv run python 03_load_model.py
```

### 4. データセット整形

読み込んだデータセットを会話形式に変換します。各サンプルを「システムプロンプト」「ユーザー入力（テキスト）」「アシスタント出力（絵文字）」の3つの役割に分けて整形し、訓練用とテスト用に分割（90%/10%）します。

**出力**: `training_dataset_splits.pkl`

```bash
uv run python 04_format_dataset.py
```

### 5. ベースモデルテスト（オプション）

ファインチューニング前のベースモデルの性能を確認します。テストデータセットからランダムにサンプルを選び、「テキストを絵文字に翻訳」というタスクに対する出力を生成して表示します。

```bash
uv run python 05_test_base_model.py
```

### 6. トレーニング設定

ファインチューニングの設定を行います：

- **BitsAndBytesConfig**: モデルを4bit量子化してメモリ効率を向上
- **LoraConfig**: パラメータ効率的なファインチューニング（PEFT）の設定
- **SFTConfig**: 教師あり学習の設定（エポック数、バッチサイズ、学習率など）

**出力**: `training_config.pkl`

```bash
uv run python 06_configure_training.py
```

### 7. モデル訓練（T4 GPUで約10分）

SFTTrainerを使ってモデルをファインチューニングします。LoRAアダプターを訓練し、各エポック終了時にチェックポイントを保存します。訓練中はTensorBoardでメトリクスを記録します。

**出力**: `myemoji-gemma-adapters/` (LoRAアダプター), `trainer.pkl`

```bash
uv run python 07_train_model.py
```

### 8. 訓練結果の可視化

訓練ログから訓練損失と検証損失を抽出し、エポックごとのグラフを作成します。過学習や未学習の状態を判断するために、両方の損失曲線を比較できます。

**出力**: `training_loss.png`

```bash
uv run python 08_plot_results.py
```

### 9. アダプターのマージ

訓練したLoRAアダプターをベースモデルとマージして、単一の完全なモデルを作成します。

**出力**: `myemoji-gemma-merged/` (マージ済みモデル), `merged_model_path.pkl`

```bash
uv run python 09_merge_adapters.py
```

### 10. ファインチューニング済みモデルのテスト

ファインチューニング後のモデルとベースモデルの出力を比較します。複数のテストプロンプト（「let's go to the beach」「I love pizza」など）に対して両方のモデルの出力を表示し、性能の改善を確認します。

```bash
uv run python 10_test_finetuned.py
```

### 11. tokenizer.model を取得

```bash
uv run 11_get_tokenizer_model.py
```

### 1２. Hugging Face Hubへアップロード（オプション）

../llama.cpp がある前提

```bash
uv run 12_quantize_gguf.py --llama-cpp-dir ../llama.cpp
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
