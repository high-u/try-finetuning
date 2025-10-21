# Hugging Face Transformers と QloRA を使用して Gemma をファインチューニングする

このガイドでは、Hugging Face の Transformers と TRL を使用して、カスタム Text-to-SQL データセットで Gemma をファインチューニングする方法について説明します。学習内容:

    量子化低ランク適応（QLoRA）とは
    開発環境を設定する
    ファインチューニング データセットを作成して準備する
    TRL と SFTTrainer を使用して Gemma をファインチューニングする
    モデル推論をテストして SQL クエリを生成する

注: このガイドは、16 GB の NVIDIA T4 GPU と Gemma 1B を使用して Google Colaboratory アカウントで実行するように作成されていますが、より大きな GPU とより大きなモデルで実行するように調整できます。
量子化低ランク適応（QLoRA）とは

このガイドでは、Quantized Low-Rank Adaptation（QLoRA） の使用方法について説明します。QLoRA は、高いパフォーマンスを維持しながら計算リソースの要件を削減できるため、LLM を効率的にファインチューニングするための一般的な方法として登場しました。QloRA では、事前トレーニング済みモデルは 4 ビットに量子化され、重みは固定されます。次に、トレーニング可能なアダプタレイヤ（LoRA）が接続され、アダプタレイヤのみがトレーニングされます。その後、アダプタの重みはベースモデルと統合することも、別個のアダプタとして保持することもできます。
開発環境を設定する
まず、TRL などの Hugging Face ライブラリと、さまざまな RLHF とアライメント手法を含むオープンモデルを微調整するためのデータセットをインストールします。

```python
# Install Pytorch & other libraries
%pip install "torch>=2.4.0" tensorboard

# Install Gemma release branch from Hugging Face
%pip install "transformers>=4.51.3"

# Install Hugging Face libraries
%pip install  --upgrade \
  "datasets==3.3.2" \
  "accelerate==1.4.0" \
  "evaluate==0.4.3" \
  "bitsandbytes==0.45.3" \
  "trl==0.15.2" \
  "peft==0.14.0" \
  protobuf \
  sentencepiece

# COMMENT IN: if you are running on a GPU that supports BF16 data type and flash attn, such as NVIDIA L4 or NVIDIA A100
#% pip install flash-attn
```

注: Ampere アーキテクチャの GPU（NVIDIA L4 など）以降を使用している場合は、フラッシュ アテンションを使用できます。Flash Attention は、計算を大幅に高速化し、メモリ使用量をシーケンス長の 2 乗から 1 乗に減らす方法です。これにより、トレーニングを最大 3 倍高速化できます。詳しくは、FlashAttention をご覧ください。

トレーニングを開始する前に、Gemma の利用規約に同意する必要があります。Hugging Face でライセンスに同意するには、モデルページ（http://huggingface.co/google/gemma-3-1b-pt）で [同意してリポジトリにアクセス] ボタンをクリックします。

ライセンスを承認した後、モデルにアクセスするには有効な Hugging Face トークンが必要です。Google Colab 内で実行している場合は、Colab のシークレットを使用することで Hugging Face トークンを安全に使用できます。それ以外の場合は、login メソッドでトークンを直接設定できます。トレーニング中にモデルを Hub に push するため、トークンに書き込みアクセス権があることを確認してください。

```python
from google.colab import userdata
from huggingface_hub import login

# Login into Hugging Face Hub
hf_token = userdata.get('HF_TOKEN') # If you are running inside a Google Colab
login(hf_token)
```

ファインチューニング データセットを作成して準備する

LLM をファインチューニングする際は、ユースケースと解決するタスクを把握することが重要です。これにより、モデルを微調整するためのデータセットを作成できます。ユースケースをまだ定義していない場合は、最初からやり直すことをおすすめします。

このガイドでは、次のユースケースを例に説明します。

    自然言語から SQL へのモデルをファインチューニングして、データ分析ツールにシームレスに統合します。目的は、SQL クエリの生成に必要な時間と専門知識を大幅に削減し、技術系以外のユーザーでもデータから有意な分析情報を抽出できるようにすることです。

テキストから SQL への変換は、データと SQL 言語に関する多くの（内部）知識を必要とする複雑なタスクであるため、LLM のファインチューニングに適したユースケースです。

ファインチューニングが適切なソリューションであると判断したら、ファインチューニングするデータセットが必要です。データセットは、解決するタスクのデモをさまざまな形で収集したものである必要があります。このようなデータセットを作成するには、次のような方法があります。

    Spider などの既存のオープンソース データセットを使用する
    LLM によって作成された合成データセットを使用する（Alpaca など）
    Dolly など、人間が作成したデータセットを使用する。
    Orca などの方法を組み合わせて使用

それぞれの方法には独自のメリットとデメリットがあり、予算、時間、品質の要件によって異なります。たとえば、既存のデータセットを使用する方法は最も簡単ですが、特定のユースケースに合わせて調整できない場合があります。一方、ドメイン エキスパートを使用する方法は最も正確ですが、時間と費用がかかる場合があります。Orca: Progressive Learning from Complex Explanation Traces of GPT-4 に示すように、複数の方法を組み合わせて命令データセットを作成することもできます。

このガイドでは、既存のデータセット（philschmid/gretel-synthetic-text-to-sql）を使用します。これは、自然言語による指示、スキーマ定義、推論、対応する SQL クエリを含む高品質の合成 Text-to-SQL データセットです。

Hugging Face TRL は、会話データセット形式の自動テンプレート化をサポートしています。つまり、データセットを適切な JSON オブジェクトに変換するだけで、trl がテンプレート化して適切な形式にします。

```json
{"messages": [{"role": "system", "content": "You are..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
{"messages": [{"role": "system", "content": "You are..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
{"messages": [{"role": "system", "content": "You are..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

philschmid/gretel-synthetic-text-to-sql には 10 万を超えるサンプルが含まれています。ガイドのサイズを小さくするため、10,000 個のサンプルのみを使用するようにダウンサンプリングされています。

これで、Hugging Face Datasets ライブラリを使用してデータセットを読み込み、プロンプト テンプレートを作成して、自然言語指示とスキーマ定義を組み合わせ、アシスタントのシステム メッセージを追加できます。

```python
from datasets import load_dataset

# System message for the assistant
system_message = """You are a text to SQL query translator. Users will ask you questions in English and you will generate a SQL query based on the provided SCHEMA."""

# User prompt that combines the user query and the schema
user_prompt = """Given the <USER_QUERY> and the <SCHEMA>, generate the corresponding SQL command to retrieve the desired data, considering the query's syntax, semantics, and schema constraints.

<SCHEMA>
{context}
</SCHEMA>

<USER_QUERY>
{question}
</USER_QUERY>
"""
def create_conversation(sample):
  return {
    "messages": [
      # {"role": "system", "content": system_message},
      {"role": "user", "content": user_prompt.format(question=sample["sql_prompt"], context=sample["sql_context"])},
      {"role": "assistant", "content": sample["sql"]}
    ]
  }

# Load dataset from the hub
dataset = load_dataset("philschmid/gretel-synthetic-text-to-sql", split="train")
dataset = dataset.shuffle().select(range(12500))

# Convert dataset to OAI messages
dataset = dataset.map(create_conversation, remove_columns=dataset.features,batched=False)
# split dataset into 10,000 training samples and 2,500 test samples
dataset = dataset.train_test_split(test_size=2500/12500)

# Print formatted user prompt
print(dataset["train"][345]["messages"][1]["content"])
```

TRL と SFTTrainer を使用して Gemma をファインチューニングする

これで、モデルをファインチューニングする準備が整いました。Hugging Face TRL の SFTTrainer を使用すると、オープン LLM を教師ありでファインチューニングできます。SFTTrainer は transformers ライブラリの Trainer のサブクラスであり、ロギング、評価、チェックポイントなど、同じ機能をすべてサポートしていますが、次のような QOL 機能も追加されています。

    データセットのフォーマット（会話形式や指示形式など）
    完了のみをトレーニングし、プロンプトを無視する
    データセットを圧縮してトレーニングを効率化する
    QloRA を含むパラメータ エフィシエント ファインチューニング（PEFT）のサポート
    会話のファインチューニング用にモデルとトークン化ツールを準備する（特殊トークンの追加など）

次のコードは、Hugging Face から Gemma モデルとトークン化ツールを読み込み、量子化構成を初期化します。

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForImageTextToText, BitsAndBytesConfig

# Hugging Face model id
model_id = "google/gemma-3-1b-pt" # or `google/gemma-3-4b-pt`, `google/gemma-3-12b-pt`, `google/gemma-3-27b-pt`

# Select model class based on id
if model_id == "google/gemma-3-1b-pt":
    model_class = AutoModelForCausalLM
else:
    model_class = AutoModelForImageTextToText

# Check if GPU benefits from bfloat16
if torch.cuda.get_device_capability()[0] >= 8:
    torch_dtype = torch.bfloat16
else:
    torch_dtype = torch.float16

# Define model init arguments
model_kwargs = dict(
    attn_implementation="eager", # Use "flash_attention_2" when running on Ampere or newer GPU
    torch_dtype=torch_dtype, # What torch dtype to use, defaults to auto
    device_map="auto", # Let torch decide how to load the model
)

# BitsAndBytesConfig: Enables 4-bit quantization to reduce model size/memory usage
model_kwargs["quantization_config"] = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=model_kwargs['torch_dtype'],
    bnb_4bit_quant_storage=model_kwargs['torch_dtype'],
)

# Load model and tokenizer
model = model_class.from_pretrained(model_id, **model_kwargs)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it") # Load the Instruction Tokenizer to use the official Gemma template
```

SFTTrainer は peft とのネイティブな統合をサポートしているため、QLoRA を使用して LLM を効率的にチューニングできます。LoraConfig を作成してトレーナーに提供するだけです。

```python
from peft import LoraConfig

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=16,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM",
    modules_to_save=["lm_head", "embed_tokens"] # make sure to save the lm_head and embed_tokens as you train the special tokens
)
```

トレーニングを開始する前に、SFTConfig インスタンスで使用するハイパーパラメータを定義する必要があります。

```python
from trl import SFTConfig

args = SFTConfig(
    output_dir="gemma-text-to-sql",         # directory to save and repository id
    max_seq_length=512,                     # max sequence length for model and packing of the dataset
    packing=True,                           # Groups multiple samples in the dataset into a single sequence
    num_train_epochs=3,                     # number of training epochs
    per_device_train_batch_size=1,          # batch size per device during training
    gradient_accumulation_steps=4,          # number of steps before performing a backward/update pass
    gradient_checkpointing=True,            # use gradient checkpointing to save memory
    optim="adamw_torch_fused",              # use fused adamw optimizer
    logging_steps=10,                       # log every 10 steps
    save_strategy="epoch",                  # save checkpoint every epoch
    learning_rate=2e-4,                     # learning rate, based on QLoRA paper
    fp16=True if torch_dtype == torch.float16 else False,   # use float16 precision
    bf16=True if torch_dtype == torch.bfloat16 else False,   # use bfloat16 precision
    max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
    warmup_ratio=0.03,                      # warmup ratio based on QLoRA paper
    lr_scheduler_type="constant",           # use constant learning rate scheduler
    push_to_hub=True,                       # push model to hub
    report_to="tensorboard",                # report metrics to tensorboard
    dataset_kwargs={
        "add_special_tokens": False, # We template with special tokens
        "append_concat_token": True, # Add EOS token as separator token between examples
    }
)
```

これで、SFTTrainer を作成してモデルのトレーニングを開始するために必要なすべてのビルディング ブロックが揃いました。

```python
from trl import SFTTrainer

# Create Trainer object
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    peft_config=peft_config,
    processing_class=tokenizer
)
```

train() メソッドを呼び出してトレーニングを開始します。

```python
# Start training, the model will be automatically saved to the Hub and the output directory
trainer.train()

# Save the final model again to the Hugging Face Hub
trainer.save_model()
```

モデルをテストする前に、メモリを解放してください。

```python
# free the memory again
del model
del trainer
torch.cuda.empty_cache()
```

QLoRA を使用する場合は、アダプターのみをトレーニングし、モデル全体はトレーニングしません。つまり、トレーニング中にモデルを保存する場合は、アダプターの重みのみを保存し、モデル全体は保存しません。完全なモデルを保存して、vLLM や TGI などのサービング スタックで簡単に使用できるようにする場合は、merge_and_unload メソッドを使用してアダプター重みをモデル重みに統合し、save_pretrained メソッドでモデルを保存します。これにより、推論に使用できるデフォルト モデルが保存されます。

注: アダプターをモデルに統合する場合は、30 GB を超える CPU メモリが必要です。この手順はスキップして、モデル推論のテストに進んでください。

```python
from peft import PeftModel

# Load Model base model
model = model_class.from_pretrained(model_id, low_cpu_mem_usage=True)

# Merge LoRA and base model and save
peft_model = PeftModel.from_pretrained(model, args.output_dir)
merged_model = peft_model.merge_and_unload()
merged_model.save_pretrained("merged_model", safe_serialization=True, max_shard_size="2GB")

processor = AutoTokenizer.from_pretrained(args.output_dir)
processor.save_pretrained("merged_model")
```

モデル推論をテストして SQL クエリを生成する
トレーニングが完了したら、モデルを評価してテストする必要があります。テスト データセットからさまざまなサンプルを読み込み、それらのサンプルでモデルを評価できます。

注: 1 つの入力に複数の正しい出力がある可能性があるため、生成 AI モデルの評価は簡単なタスクではありません。このガイドでは、手動評価とバイブチェックのみについて説明します。

```python
import torch
from transformers import pipeline

model_id = "gemma-text-to-sql"

# Load Model with PEFT adapter
model = model_class.from_pretrained(
  model_id,
  device_map="auto",
  torch_dtype=torch_dtype,
  attn_implementation="eager",
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
```

テストデータセットからランダムなサンプルを読み込み、SQL コマンドを生成しましょう。

```python
from random import randint
import re

# Load the model and tokenizer into the pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Load a random sample from the test dataset
rand_idx = randint(0, len(dataset["test"]))
test_sample = dataset["test"][rand_idx]

# Convert as test example into a prompt with the Gemma template
stop_token_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<end_of_turn>")]
prompt = pipe.tokenizer.apply_chat_template(test_sample["messages"][:2], tokenize=False, add_generation_prompt=True)

# Generate our SQL query.
outputs = pipe(prompt, max_new_tokens=256, do_sample=False, temperature=0.1, top_k=50, top_p=0.1, eos_token_id=stop_token_ids, disable_compile=True)

# Extract the user query and original answer
print(f"Context:\n", re.search(r'<SCHEMA>\n(.*?)\n</SCHEMA>', test_sample['messages'][0]['content'], re.DOTALL).group(1).strip())
print(f"Query:\n", re.search(r'<USER_QUERY>\n(.*?)\n</USER_QUERY>', test_sample['messages'][0]['content'], re.DOTALL).group(1).strip())
print(f"Original Answer:\n{test_sample['messages'][1]['content']}")
print(f"Generated Answer:\n{outputs[0]['generated_text'][len(prompt):].strip()}")
```
