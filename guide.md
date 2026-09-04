1. プロジェクトを開く

VS Codeで、

C:\Users\hirok\Downloads\s-ai-main\s-ai-main

を開きます。

最終的に、ターミナルで

cd C:\Users\hirok\Downloads\s-ai-main\s-ai-main

となっている状態にします。


---

2. Pythonを確認

ターミナルで、

python --version

または

py --version

を実行。

Python 3.10～3.12程度を推奨します。


---

3. 仮想環境を作成

プロジェクトフォルダで、

python -m venv .venv

作成後、

.venv\Scripts\activate

成功するとターミナルの先頭に、

(.venv)

のように表示されます。

もしPowerShellの実行ポリシーで止まったら、

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

を実行してから、

.venv\Scripts\activate

です。


---

4. 必要なライブラリをインストール

まずPyTorch。

NVIDIA GPUを使う場合

CUDA対応版を入れます。

pip install torch torchvision torchaudio

CPUだけの場合

pip install torch torchvision torchaudio

基本的にはこれでOKです。

その後、

pip install fastapi uvicorn pydantic

も入れます。

まとめて、

pip install torch torchvision torchaudio fastapi uvicorn pydantic


---

5. PyTorchがGPUを認識しているか確認

python -c "import torch; print('PyTorch:',torch.__version__); print('CUDA:',torch.cuda.is_available()); print('GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

例えば、

PyTorch: 2.x.x
CUDA: True
GPU: NVIDIA ...

ならGPUで実行できます。

CUDA: FalseならCPU実行になります。


---

6. tokenizerを作成

プロジェクトフォルダで、

python tokenizer.py

実行します。

成功すると、

tokenizer/tokenizer.json

が作成されます。

最後に、

ALL TOKENIZER TESTS PASSED

が表示されればOKです。


---

7. 学習データを確認

最低限、

data/
├── pretrain/
│   └── example.txt
└── sft/
    └── train.jsonl

が存在することを確認します。

SFTデータは、

{"messages":[{"role":"user","content":"こんにちは"},{"role":"assistant","content":"こんにちは！"}]}

のような形式です。


---

8. まずPretrainを実行

python pretrain.py

これで、

data/pretrain/

のデータを使って事前学習します。

学習が進むと、

checkpoints/latest.pt

や、

checkpoints/pretrain_final.pt

などが作られます。

注意

現在の設定では、

max_steps = 100000

なので、かなり長く学習する設定です。

動作確認だけなら、まず config.py の

max_steps=100000

を例えば、

max_steps=100

程度にして実行するのがおすすめです。

本格的に学習するときに戻します。


---

9. SFTを実行

Pretrainが終わったら、

python posttrain.py

を実行します。

これで、

data/sft/train.jsonl

を使って会話形式に調整します。

成功すると、

checkpoints/sft_latest.pt

と、

checkpoints/sft_final.pt

が作られます。


---

10. CLIでNovaLLMをテスト

SFTが完了したら、

python inference.py

を実行。

すると、

======================================================================
NovaLLM interactive mode
終了: exit / quit
======================================================================

You >

のようになります。

例えば、

You > こんにちは

と入力します。

NovaLLMが、

Nova > こんにちは！

のように返答すれば成功です。

終了は、

exit

または、

quit

です。


---

11. Web UIを起動

CLIが正常に動いたら、

python server.py

を実行します。

成功するとサーバーが、

Uvicorn running on http://0.0.0.0:8000

のように起動します。

ブラウザで、

[http://localhost:8000](http://localhost:8000?utm_source=chatgpt.com)

を開きます。

NovaLLMのWeb UIが表示されます。


---

12. APIだけ確認する場合

ブラウザで、

[http://localhost:8000/health](http://localhost:8000/health?utm_source=chatgpt.com)

を開きます。

例えば、

{
  "status": "ok",
  "model_loaded": true,
  "device": "cuda",
  "cuda": true,
  "ui": true
}

なら正常です。

特に、

"model_loaded": true

が重要です。


---

13. 実行順序まとめ

基本的にはこの順番です。

プロジェクトを開く
        ↓
仮想環境を作る
        ↓
ライブラリをインストール
        ↓
tokenizer.py
        ↓
pretrain.py
        ↓
posttrain.py
        ↓
inference.py
        ↓
server.py
        ↓
http://localhost:8000

つまり初回は、

cd C:\Users\hirok\Downloads\s-ai-main\s-ai-main
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision torchaudio fastapi uvicorn pydantic
python tokenizer.py
python pretrain.py
python posttrain.py
python inference.py

までやって、CLIで正常動作を確認。

その後、

python server.py

でWeb版を起動します。


---

14. 2回目以降

すでに学習済みなら、毎回Pretrainからやる必要はありません。

cd C:\Users\hirok\Downloads\s-ai-main\s-ai-main
.venv\Scripts\activate
python server.py

だけでOKです。

CLIなら、

python inference.py

です。


---

15. 重要なファイル

最終的に特に重要なのはこの4つです。

tokenizer/tokenizer.json
checkpoints/pretrain_final.pt
checkpoints/sft_final.pt
data/sft/train.jsonl

特に現在のNovaLLMは語彙数264の新しいTokenizerを使っているため、以前作った古い260語彙のチェックポイントをそのまま使わないでください。


---

一番おすすめの確認方法

いきなり10万stepを回すより、

① tokenizer.py
② pretrain.pyを少ないstepでテスト
③ posttrain.pyを少ないstepでテスト
④ inference.py
⑤ server.py
⑥ Web UI
⑦ 問題なければ本格学習

の順番にすると、エラーを見つけやすいです。
