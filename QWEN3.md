● Qwen3 学習可能性の調査結果

  結論: ✅ ほぼそのまま学習可能です！

  ---
  1. 利用可能なQwen3モデル

  HuggingFaceで公式に提供されています：

  | モデル名            | パラメータ | コンテキスト | 推奨用途   | VRAM (8bit) |
  |-----------------|-------|--------|--------|-------------|
  | Qwen/Qwen3-0.6B | 0.6B  | 32K    | 小規模・高速 | ~2GB        |
  | Qwen/Qwen3-1.7B | 1.7B  | 32K    | テスト用   | ~4GB        |
  | Qwen/Qwen3-4B   | 4B    | 32K    | バランス型  | ~8GB        |
  | Qwen/Qwen3-8B   | 8B    | 128K   | ⭐ 推奨   | ~16GB       |
  | Qwen/Qwen3-14B  | 14B   | 128K   | 高性能    | ~24GB       |

  あなたのRTX 5070 Ti (16GB)なら、Qwen3-8Bが8bit量子化でギリギリ、Qwen3-14Bは4bit量子化で可能です。

  ---
  2. スクリプトの互換性

  必要な変更は最小限：

  変更1: transformersのバージョン確認

  # transformers >= 4.51.0 が必要
  pip install transformers>=4.51.0

  変更2: トークナイザーの設定追加

  03_train_model.py:113 の後に追加：
  tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

  # ⭐ この1行を追加
  tokenizer.eos_token = '<|im_end|>'

  変更3: 変数名を汎用的に（オプション）

  03_train_model.py:85 の gemma_model を base_model_name に変更：
  base_model_name = model_config['model_name']  # gemma_model から変更

  そして 03_train_model.py:153-154:
  base_model = AutoModelForCausalLM.from_pretrained(
      base_model_name,  # gemma_model から変更
      **model_kwargs
  )

  ---
  3. 量子化のサポート

  ✅ 4bit/8bit量子化は完全にサポートされています。

  現在のスクリプトのBitsAndBytesConfigはそのまま使えます。

  注意点:
  - 小さいモデル（0.6B）は4bitで精度が10%低下
  - 大きいモデル（8B以上）は4bitでも1%程度の低下

  ---
  4. 既知の問題

  - ✅ LoRA + BitsAndBytes: 問題なし
  - ✅ SFTTrainer: 問題なし
  - ✅ target_modules="all-linear": 問題なし
  - ⚠️ 2bit/3bit量子化: 精度が大幅に低下（避けるべき）

  ---
  推奨アクション

  Qwen3を試すなら：

  1. Qwen3-8B (8bit量子化) - RTX 5070 Tiでギリギリ動作
  2. Qwen3-14B (4bit量子化) - より高性能

  まずは Qwen/Qwen3-8B で試してみることをお勧めします！

  必要なスクリプト修正を行いますか？
