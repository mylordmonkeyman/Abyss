# Abyss マニュアル

Abyss のビルド、DeepSeek V4 Flash の準備、起動と調整の手順。English: [MANUAL.md](MANUAL.md)

Abyss は [Project Maya](https://github.com/mw00/project-maya)（MIT）をコピーして作ったフォークで、Maya は
[Strata](https://github.com/Niko1221/Strata)（MIT）から生まれたプロジェクト。エンジン、エキスパートの配置の仕組み、
サーバー、ダッシュボードは Maya のもので、Abyss はそこに DeepSeek V4 への対応を足した。Maya の元のドキュメント
（GLM-5.3-Flash、AMD、複数 GPU、インストーラ）は [docs/maya/](maya/README.md) にそのまま残している。

- [必要なもの](#必要なもの)
- [まとめてセットアップ](#まとめてセットアップ)
- [ビルド](#ビルド)
- [モデルを取る](#モデルを取る)
- [pack を作る](#pack-を作る)
- [設定と起動](#設定と起動)
- [使い方](#使い方)
- [メモリと速度](#メモリと速度)
- [設定項目](#設定項目)
- [llama.cpp との照合](#llamacpp-との照合)
- [うまくいかないとき](#うまくいかないとき)
- [制限](#制限)

## 必要なもの

| | 確認した環境 | 補足 |
| --- | --- | --- |
| OS | Ubuntu (Linux) | Windows は DeepSeek V4 では未確認 |
| GPU | RTX 5090 (32 GB) | GPU のアーキテクチャに合わせてビルドする（RTX 50 は `120a`）。ほかの NVIDIA GPU もビルドはできるが DeepSeek V4 では未確認。Blackwell より前の GPU では MXFP4 のエキスパートは FP4 MMA ではなく q8_1 の経路で計算する |
| RAM | 94 GB | モデルは 96.8 GB。VRAM + RAM が約 100 GB を超えれば全エキスパートがメモリに載る。足りない分は SSD から読む（動くが遅い） |
| ディスク | NVMe SSD、空き約 100 GB | GGUF ファイル。pack は 0.2 GB |
| CUDA | 13.3 | `nvcc` を含む |
| コンパイラ | g++ 13 | CUDA が受け付けるバージョン |
| Python | 3.12 | Abyss のフォルダ内の `.venv` |
| ROCm（任意） | 7（`hipcc`） | AMD APU の内蔵 GPU が RAM 層のエキスパートを計算する（下の「iGPU レーン」）。なければ CPU が計算する |

`./abyss.sh`（Maya のインストーラの名前を変えたもの）は、今も Maya の GLM-5.3-Flash を入れるためのもの。
DeepSeek V4 は `./setup-deepseek.sh`（次の節）か、その後の手順を手で行って準備する。

## まとめてセットアップ

```sh
git clone https://github.com/zephel01/Abyss.git ~/llm/apps/Abyss
cd ~/llm/apps/Abyss
./setup-deepseek.sh --download-model
```

`setup-deepseek.sh` は `.venv` を作ってから `tools/abyss_setup.py` を実行する。済んだ手順は飛ばすので、何度実行してもよい:

1. PC の確認（NVIDIA GPU、RAM）
2. Python パッケージ、固定コミットの llama.cpp のソース
3. GPU に合わせてエンジンをビルド（RTX 50 は `120a`）
4. モデルを `../Abyss-data/models/unsloth/DeepSeek-V4-Flash-0731-GGUF/<量子化>` にダウンロード（`--download-model` が
   なければ先に確認する）。全ファイルを Hugging Face の SHA-256 と照合する
5. シャードと同じフォルダに pack を作る
6. Abyss のフォルダに設定 `abyss-<名前>.json` と起動スクリプト `run-abyss-<名前>.sh` を書く
7. Abyss の隣に **Abyss-tools** フォルダを作る（`abyss.sh start | stop | restart | status | update | build | log | models`。
   雛形は本体の `ops/abyss-tools/`）。セットアップを再実行すると `abyss.sh` / `common.sh` は雛形に合わせて更新し
   （前のものは `.bak` に残す）、自分で書いた `README.md` はそのまま残す

```sh
../Abyss-tools/abyss.sh start      # 起動後 http://127.0.0.1:8082/
../Abyss-tools/abyss.sh stop
```

| オプション | |
| --- | --- |
| `--quant UD-IQ3_XXS` | unsloth/DeepSeek-V4-Flash-0731-GGUF のほかの量子化（既定 `UD-Q2_K_XL`。確認済みはこれだけ） |
| `--name 名前` | `abyss-<名前>.json` の名前（既定 `dsv4-<量子化>`。例 `dsv4-q2_k_xl`） |
| `--context N`, `--port N` | コンテキスト長（既定 65536）とポート（既定 8082） |
| `--data-dir DIR`, `--tools-dir DIR` | Abyss-data と Abyss-tools を別の場所にする |
| `--gpu N`, `--arch A` | 使う GPU、コンパイルする CUDA アーキテクチャ |
| `--download-model`, `--yes` | 確認せずにダウンロード、すべての質問に既定値で答える |
| `--rewrite-config` | `abyss-<名前>.json` を書き直す（なければ既存のものを残す） |
| `--tools-only` | Abyss-tools だけを作る / 更新する |

以下の節は、同じことを手で行う手順。

## ビルド

```sh
git clone https://github.com/zephel01/Abyss.git ~/llm/apps/Abyss
cd ~/llm/apps/Abyss

# このフォルダ専用の Python 環境
python3 -m venv .venv
.venv/bin/pip install numpy jinja2 regex pyyaml tqdm requests cmake ninja pillow psutil

# エンジンが固定している llama.cpp のソース（setup.py の LLAMA_CPP_COMMIT）: ggml、MMQ カーネル、gguf-py
C=3cf03257f219afbe7334045ff7c6a06ac68c627d
curl -L -o /tmp/llama.zip https://github.com/ggml-org/llama.cpp/archive/$C.zip
unzip -q /tmp/llama.zip -d third_party && mv third_party/llama.cpp-$C third_party/llama.cpp

# エンジン: build/strata
.venv/bin/cmake -S . -B build -G Ninja -DCMAKE_MAKE_PROGRAM=$PWD/.venv/bin/ninja -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120a -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_BUILD_TESTS=OFF \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DSTRATA_GGML_DIR=$PWD/third_party/llama.cpp
.venv/bin/cmake --build build --target strata -j 16
```

ROCm の `hipcc` が入っていれば、`build/libabyss_igpu.so`（下の「iGPU レーン」）も一緒にビルドされる
（対象は既定で `-DSTRATA_IGPU_ARCH=gfx1151`。`-DSTRATA_IGPU_LANE=OFF` で作らない）。

初回のビルドは数分。変更後は変わった部分だけを再コンパイルする。RTX 50 では `120` ではなく `120a` を使う
（MXFP4 のカーネルが Blackwell 専用の FP4 MMA 命令を使い、`120` ではコンパイルできない）。

## モデルを取る

unsloth の [DeepSeek-V4-Flash-0731-GGUF](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF) の `UD-Q2_K_XL`
（3 ファイル、96.8 GB）。Hugging Face の CLI（`pip install huggingface_hub` で `hf` が入る）で取る:

```sh
M=~/llm/apps/Abyss-data/models/unsloth/DeepSeek-V4-Flash-0731-GGUF   # Abyss の隣の Abyss-data（NVMe ならどこでもよい）
hf download unsloth/DeepSeek-V4-Flash-0731-GGUF \
  UD-Q2_K_XL/DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00001-of-00003.gguf \
  UD-Q2_K_XL/DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00002-of-00003.gguf \
  UD-Q2_K_XL/DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00003-of-00003.gguf --local-dir $M
sha256sum $M/UD-Q2_K_XL/*.gguf
```

| ファイル | SHA-256 |
| --- | --- |
| `-00001-of-00003.gguf` (5 MB) | `de65c5bb660817b95cc281bf54935f9d1bcc3b22535f2eae63b35a14c7c5724b` |
| `-00002-of-00003.gguf` (49.4 GB) | `c8728a298862cd8736782e2e4056e5e3e03c415ba552a38173264494e41ef327` |
| `-00003-of-00003.gguf` (47.4 GB) | `a4bae0788c025c1b2388754592c98fefa1372981748ac9875039bf6a30e6eeb4` |

ダウンロードが途中で止まって再開しないときは、
`curl -L -C - -o <ファイル> https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF/resolve/main/UD-Q2_K_XL/<ファイル>`
で続きから取れる。

同じリポジトリのほかの量子化も、エキスパートの型が decode 用カーネルの対応範囲（IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS,
IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS, Q2_K〜Q6_K, MXFP4。IQ1_M はまとめ処理のカーネルがないので prompt は 1 トークンずつ）
なら動くはず。確認したのは UD-Q2_K_XL だけ。

## pack を作る

pack はエンジンが使う GGUF の索引（小さいテンソルの dense.bin、エキスパートの表、チャットテンプレート付きの
トークナイザー）。シャードと同じフォルダに作る:

```sh
cd ~/llm/apps/Abyss
.venv/bin/python tools/iq_pack.py --gguf $M/UD-Q2_K_XL/DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00001-of-00003.gguf \
  --out $M/UD-Q2_K_XL/pack-dsv4 --compat-bf16
```

`--compat-bf16` は、GGUF に F32 で入っているハイパーコネクションの行列を、エンジンが読む BF16 に丸める。数秒で終わる。
エキスパートは GGUF のまま使う。

## 設定と起動

設定ファイル（名前は自由。ここでは `strata-dsv4.json`）を Abyss のフォルダに置く:

```json
{
 "exe": "/home/you/llm/apps/Abyss/build/strata",
 "args": ["--glm-pack", "/home/you/llm/apps/Abyss-data/models/unsloth/DeepSeek-V4-Flash-0731-GGUF/UD-Q2_K_XL/pack-dsv4",
          "--max-context", "65536", "--prefill", "auto"],
 "cwd": "/home/you/llm/apps/Abyss",
 "tokenizer": "/home/you/llm/apps/Abyss-data/models/unsloth/DeepSeek-V4-Flash-0731-GGUF/UD-Q2_K_XL/pack-dsv4/tokenizer",
 "model_name": "deepseek-v4-flash",
 "sampling": {"temperature": 1.0, "top_p": 0.95},
 "port": 8082,
 "log": "/home/you/llm/apps/Abyss/strata-dsv4.log"
}
```

| 項目 | |
| --- | --- |
| `exe` | エンジン（`build/strata`） |
| `args` | エンジンの引数。`--glm-pack` に pack（DeepSeek V4 も GLM と同じ入口で動く）、`--max-context` にコンテキスト長、`--prefill auto` は prompt のまとめ処理の大きさ |
| `tokenizer` | pack の `tokenizer` フォルダ（チャットテンプレートは DeepSeek のもの） |
| `model_name` | API が返すモデル名 |
| `sampling` | リクエストが指定しないときの既定値（DeepSeek の推奨: temperature 1.0, top_p 0.95） |
| `reasoning_effort` | 省略可。思考の既定レベル（`none`, `low`, `high`, `max`） |
| `thinking_budget` | 省略可。思考の上限トークン数（既定 32768、0 で上限なし） |
| `env` | 省略可。エンジンの設定 `{"STRATA_GLM_...": "..."}`（[設定項目](#設定項目)） |
| `log` | エンジンのログ |

サーバーを起動する:

```sh
.venv/bin/python serve/server.py --engine strata --config strata-dsv4.json --port 8082
```

読み込みが終わると `ready: http://127.0.0.1:8082/v1` と表示される。起動には 1 分ほどかかる（密な重みを GPU に載せ、
全エキスパートを VRAM と RAM に読み込む。NVMe から 4 GB/s で約 20 秒）。Ctrl+C で止まる。

サーバーのオプション: `--host 0.0.0.0` でほかのマシンからの接続を受け付ける（必ず `--api-key <secret>` も付ける）、
`--open` でブラウザを開く、`--gpu 0` で GPU を選ぶ。

## 使い方

- **ブラウザ:** `http://127.0.0.1:8082/` でチャット、エキスパートの配置と GPU のモニター、設定。
- **OpenAI 互換 API:** ベース URL `http://127.0.0.1:8082/v1`、モデル名は何でもよい。

  ```sh
  curl http://127.0.0.1:8082/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"What is 17*23?"}],"max_tokens":4000}'
  ```

  回答は `content`、思考は `reasoning_content` に入る。
- **Anthropic 互換 API:** `http://127.0.0.1:8082/v1/messages`（Claude Code なら `ANTHROPIC_BASE_URL=http://127.0.0.1:8082`）。
- **思考:** DeepSeek V4 は指定しなければ考えてから答える。`reasoning_effort`（またはチャットの設定）で変えられる:

  | レベル | DeepSeek に渡るもの |
  | --- | --- |
  | 指定なし | 思考あり |
  | `none` | 思考なし（prompt の末尾が `</think>`） |
  | `low` | 思考あり |
  | `high` | 思考あり + DeepSeek の「Reasoning Effort: Absolute maximum」の指示 |
  | `max` | 思考あり + 「Beyond maximum」の指示 |

  サーバーは回答に最低 1024 トークン（または `max_tokens` の 4 分の 1）を必ず残す。`max_tokens` が小さい（数百）と、
  思考はほぼすぐに打ち切られる。思考させるときは `max_tokens` を数千にする。
- **ツール呼び出し:** OpenAI の `tools`、Anthropic の `tools` が使える。1 回の応答で複数の呼び出しも可。モデルは
  DeepSeek の DSML 形式で書き、サーバーがそれを `tool_calls` に変換する。過去の呼び出しと `tool` の結果は
  prompt に描画し直す。
- **会話の再利用:** 前のリクエストの続き（同じメッセージ + 新しいメッセージ）は、新しいトークンだけを読む。
- **リクエストは 1 つずつ**処理する。ほかは順番待ち。

### Strata-tools の strata.sh を使う場合

`strata.sh`（start / stop / log）を使うなら、Abyss のフォルダに起動スクリプトと設定を `run-<名前>.sh`,
`strata-<名前>.json` の名前で置いて、`STRATA_DIR=~/llm/apps/Abyss ./strata.sh start <名前>` で起動する。
起動スクリプトの中身は上の起動コマンド（`exec .venv/bin/python serve/server.py --engine strata "--config" "<設定>" "--port" "8082" "$@"`）。

## メモリと速度

エンジンは、密な重み（UD-Q2_K_XL で 6.2 GB）、キャッシュ、prompt 用のバッファを置いた残りの VRAM をエキスパートで
埋め、空き RAM から pinned の RAM 層を確保し、足りない分は必要なときに SSD から読む。使ったエキスパートを
`<pack>/expert_usage.txt` に記録し、次の起動ではそれを先に読み込む。

RTX 5090 + Ryzen AI Max+ 395（RAM 94 GB）での実測（UD-Q2_K_XL、コンテキスト 65536、全エキスパートが
VRAM 23 GB + RAM 70 GB に載った状態、iGPU レーンあり）:

| | |
| --- | --- |
| 生成（1,500 トークン前後の応答） | 約 73 tok/s（CPU だけなら約 56） |
| 生成（コーディングの prompt 20 本、各 800 トークンまで） | 約 66 tok/s。`STRATA_DS4_SPEC=1` で約 78 |
| prompt（5.6K トークン） | 約 1,000〜1,090 tok/s |
| prompt（18K トークン） | 約 1,700 tok/s |
| 起動（重み + エキスパートの読み込み） | 約 1 分 |
| 65536 トークン分のアテンションキャッシュ | 0.44 GB |

起動直後や、メモリがモデルより少ない環境では、エキスパートを SSD から読むので遅くなる。RAM 層のエキスパートは、
一部を CPU が計算し、残りを GPU が PCIe 経由で取り寄せる。その配分は起動時に測って決める。

**iGPU レーン。** AMD の APU（Ryzen AI Max+ 395 の Radeon 8060S）では、内蔵 GPU が CPU と同じメモリを約 240 GB/s
で読める（CPU のスレッドは約 75 GB/s）。ROCm が入っていて `libabyss_igpu.so` がビルドされていれば、エンジンは
起動時に RAM 層を内蔵 GPU に登録し、RAM 層のエキスパートを CPU の代わりに内蔵 GPU で計算する（コピーはしない）。
どちらで計算するかは起動時に両方を測って決め、ログに出す（`iGPU lane: an expert 0.046 ms vs 0.107 on the CPU's
16 threads`）。CPU のほうが速い場合や、ライブラリか HIP のデバイスがない場合は CPU で計算する。

**投機デコード（`STRATA_DS4_SPEC`）。** この GGUF にはドラフトモデルがないが、応答は prompt や応答の前のほうに
出てきた並び（コード、ファイルパス、ツール呼び出し）を繰り返すことが多い。`STRATA_DS4_SPEC=1` にすると、直近
3 トークンが前に出てきた場所を探し、その次のトークンを候補として次の 1 トークンと一緒に 1 回で確かめ、モデルの
答えと一致すれば採用する。出力は使わない場合と同じで（同じロジットに同じサンプラー）、速さだけが変わる。
コーディングの prompt で +12%（候補の 77〜83% が採用）、文章ではほぼ変わらない（採用は約 50%）。

## 設定項目

エンジンへの環境変数。設定ファイルの `"env"` に書く（Maya の全項目は [docs/maya/README.md](maya/README.md#tuning)。
ここで特に関係するもの）:

| 設定 | 既定値 | |
| --- | --- | --- |
| `STRATA_GLM_WARM` | オン | `0`: 起動時に読み込まない（使われたエキスパートから載っていく。最初の応答が遅い） |
| `STRATA_GLM_RAM_GB` | 空き RAM から | pinned の RAM 層の大きさを GB で固定 |
| `STRATA_GLM_RAM_HEADROOM_GB` | 6 | RAM 層を決めるとき、システムのために空けておく RAM |
| `STRATA_GLM_RESERVE_MB` | 700 | エキスパート用プールの横に空けておく VRAM |
| `STRATA_GLM_CPU_LANE` | 物理コアごとに 1 スレッド | RAM 層のエキスパートを計算する CPU スレッド数。`0` で無効 |
| `STRATA_GLM_PCIE_SHARE` | 起動時に測定 | RAM 層のエキスパートのうち、CPU で計算せずに PCIe で GPU に送る割合（`0` = すべて CPU、`1` = すべて GPU） |
| `STRATA_GLM_PREFILL` / `--prefill` | `auto` | prompt のまとめ処理の大きさ: `auto`、トークン数、`0`（1 トークンずつ） |
| `STRATA_GLM_USAGE` | `<pack>/expert_usage.txt` | 起動をまたいで残すエキスパートの使用記録。`0` で無効 |
| `STRATA_GLM_NO_REUSE` | オフ | `1`: 会話を再利用しない |
| `STRATA_GLM_THINK_RESERVE` | 1024 | （サーバー）思考が回答のために必ず残すトークン数 |
| `STRATA_GLM_IGPU` | 自動 | iGPU レーン。`libabyss_igpu.so` と HIP のデバイスがあり、CPU より速ければ使う。`0` で使わない、`1` で遅くても使う。`STRATA_GLM_IGPU_LIB=<パス>` で別のライブラリ |
| `STRATA_DS4_SPEC` | 0（なし） | prompt と応答から作る投機デコード。1 回に試す候補トークン数（1〜3。実測では 1 が最速） |
| `STRATA_DS4_SPEC_MIN` | 3 | 候補を探すときに一致させるトークン数（2〜4） |

コンテキスト長は `--max-context` で決める。DeepSeek V4 の学習は 1M トークンだが、確認したのは 65536。アテンションの
キャッシュは小さい（65536 で 0.44 GB）が、prompt 用のバッファはコンテキストに比例して増える。

## llama.cpp との照合

`tools/dsv4_logits_dump.cpp` は、トークン列に対する llama.cpp の logits を書き出す（`deepseek4` に対応した llama.cpp に
対してビルドする）。エンジンの `STRATA_GLM_SCORE` モードで、エンジン自身の logits と比べられる。手順と結果は
[docs/deepseek4/PORT.md](deepseek4/PORT.md)。UD-Q2_K_XL では、3000 トークンで KL 0.009、次トークンの的中率も同じだった。

## うまくいかないとき

- **ビルドが `mma with block scale not supported on .target sm_120` で止まる** - `-DCMAKE_CUDA_ARCHITECTURES=120a` で
  構成し直す。
- **`the engine exited before it was ready`** - エンジンのログ（設定の `log`）を見る。RAM 層の確保でメモリが足りないなら、
  `STRATA_GLM_RAM_GB` を下げるか `STRATA_GLM_RAM_HEADROOM_GB` を上げる。
- **`refers to missing files` や `pack:` のエラー** - pack は、作ったときの 3 つのシャードと同じ場所に置く必要がある。
  シャードを移したら作り直す。
- **思考が一言で止まって回答が始まる** - `max_tokens` が小さすぎる（上の「思考」を参照）。
- **遅く、SSD がずっと動いている** - エキスパートに対して RAM + VRAM が足りない。ほかのプログラムを閉じるか、RAM を増やす。
- **ポートが使われている** - 別のサーバーが動いている。`--port` で変える。

## 制限

- ドラフトモデルはない。この GGUF には MTP ブロックがなく、DeepSeek の DSpark（ドラフトモデル）には未対応。
  投機デコードは文章自体から候補を作る（`STRATA_DS4_SPEC`）。
- 会話のディスク保存（`STRATA_GLM_SLOTS`）は DeepSeek V4 では無効。直前の会話の再利用は動く。
- 画像は扱わない（ここでは DeepSeek V4 Flash はテキストのみ）。
- 確認したのは GPU 1 枚だけ。Maya の複数 GPU への層の分割は、DeepSeek V4 では未確認。
- `./abyss.sh`（Maya から引き継いだインストーラ）は GLM 用。DeepSeek V4 は `./setup-deepseek.sh`。
