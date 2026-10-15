<div align="center">

# Abyss

**284B の MoE モデル DeepSeek V4 Flash を、家庭用の GPU 1 枚で動かす。**

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Platform](https://img.shields.io/badge/platform-Linux-lightgrey)
![CUDA](https://img.shields.io/badge/CUDA-13.x-76B900?logo=nvidia&logoColor=white)
![ROCm](https://img.shields.io/badge/ROCm-7%20(experimental)-ED1C24?logo=amd&logoColor=white)
![GPU](https://img.shields.io/badge/tested-RTX%205090%20%C2%B7%20RTX%203090%20%C2%B7%20Radeon%208060S-76B900)
![Model](https://img.shields.io/badge/model-DeepSeek--V4--Flash-4D6BFE)
[![Release](https://img.shields.io/github/v/release/zephel01/Abyss?label=release)](https://github.com/zephel01/Abyss/releases)

[English](README.md) · **日本語**

[クイックスタート](#-クイックスタート) · [性能](#-性能) · [しくみ](#-しくみ) ·
[マニュアル](docs/MANUAL.ja.md) · [更新履歴](CHANGELOG.md) · [クレジット](#-クレジット)

</div>

---

Abyss は [DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) を GGUF から NVIDIA の GPU 1 枚で
動かす。よく使うエキスパートを VRAM に、次に使うものを pinned RAM に、残りを NVMe SSD に置き、会話に合わせて入れ替える。
ブラウザのチャットと、OpenAI / Anthropic 互換の API（思考とツール呼び出しに対応）が、すべて手元のマシンだけで動く。

> [!NOTE]
> Abyss は [Project Maya](https://github.com/mw00/project-maya) をコピーして作ったフォークで、Maya は
> [Strata](https://github.com/Niko1221/Strata) から生まれたプロジェクト。Maya のエンジンに DeepSeek V4 への対応を足した。
> [クレジット](#-クレジット)を参照。

## ✨ 特長

- **284B の MoE を GPU 1 枚で** — 層ごとに 256 個あるエキスパートを VRAM・pinned RAM・SSD に分けて置く。どれを先に載せるかは使い方から学ぶ
- **生成は llama.cpp の約 3 倍、prompt の読み込みは約 5 倍**（同じ GGUF・同じマシン。llama.cpp は `-ncmoe` で調整した状態。エキスパートを CPU に置いた llama.cpp と比べると 10 倍以上）。prompt は約 1,600〜3,300 tok/s、生成は約 82〜84 tok/s
- **長いコンテキストでも落ちない** — lightning indexer と prompt の attention を tensor core で計算する。125K トークンの prompt を約 2,800 tok/s で読み、125K でも生成は約 65 tok/s
- **AMD APU の内蔵 GPU を 2 本目のレーンに** — Ryzen AI Max では Radeon の内蔵 GPU が RAM 層のエキスパートをその場で計算する。生成でも prompt でも使う（ROCm。起動時に CPU より速ければ使う）
- **ドラフトモデルなしの投機デコード** — prompt や応答に出てきた並び（n-gram）を候補にして、次のトークンと一緒に 1 回で確かめる。出力は同じで、コードで +10〜13%。既定でオン
- **llama.cpp と照合済み** — 3,000 トークンで次トークンの的中率が同じ（KL 0.009）
- **DeepSeek V4 のアテンションを全部実装** — スライディングウィンドウ、CSA/HCA 圧縮 KV、インデクサの top-512、sinks
- **思考とツール呼び出し** — 思考レベル `none` / `low` / `high` / `max`、DeepSeek の DSML 形式のツール呼び出し（同時呼び出しも可）
- **OpenAI / Anthropic 互換 API** と、エキスパートの配置をリアルタイムで見られるブラウザのダッシュボード
- **複数 GPU、AMD、V100** — NVIDIA の複数カードに層を分けて載せる。Radeon 向けの HIP ビルド（試験的）、CUDA 12 での sm_70 ビルド
- **コマンド 1 つでセットアップ** — エンジン、モデル、pack、設定、運用スクリプトの `Abyss-tools` まで

## 📊 性能

unsloth の [DeepSeek-V4-Flash-0731-GGUF](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF) `UD-Q2_K_XL`
（96.8 GB）、RTX 5090（32 GB、PCIe Gen4 x4）+ Ryzen AI Max+ 395（RAM 94 GB、iGPU レーンあり）、Linux、Abyss v1.3.16。
llama.cpp（`bed0a8566`）は同じマシン・同じファイルで、いちばん速い設定（`-ngl 99 -ncmoe 32`）:

| | Abyss | llama.cpp |
| --- | :---: | :---: |
| 生成（1,024 トークンの応答、文章 / コード） | **約 90〜93 tok/s** | 28〜30 tok/s |
| 生成（swe-bench のコーディング問題 16 問 × 3、通し） | **約 99 tok/s** | – |
| 生成（コンテキスト 125K トークン） | **約 65 tok/s** | – |
| prompt（8K トークン） | **約 1,630〜1,760 tok/s** | 78（ubatch 512）〜 660（ubatch 8192） |
| prompt（32K トークン） | **約 3,100〜3,270 tok/s** | – |
| prompt（125K トークン） | **約 2,800 tok/s** | – |
| 起動（重み + エキスパートの読み込み） | 約 1 分 | 約 50 秒 |

エキスパート 11,008 個がすべて VRAM（23 GB）+ pinned RAM（70 GB）に収まる。読み込み後は SSD を読まない。生成の数値は
attention の射影を Q5_K にした「軽い dense pack」（マニュアル参照。配布 GGUF のままより +8%、swe-bench の品質は同じ）での実測。
内蔵 GPU がなく CPU レーンで計算する環境（RAM が足りる NVIDIA の PC）では、生成は約 55〜60 tok/s。
数値はすべて測り方とともに[更新履歴](CHANGELOG.md)にある。

<details>
<summary><b>ほかに測った構成</b></summary>

| 構成 | 生成 | prompt 8K / 32K |
| --- | :---: | :---: |
| RTX 3090（USB4 ケース、約 3.8 GB/s）+ iGPU レーン | 約 49 tok/s | 約 690 / 約 1,010 tok/s |
| RTX 5090 + RTX 3090（層を 27 / 16 に分割）+ iGPU レーン | 約 55〜57 tok/s | 約 1,500 / 約 1,990 tok/s |
| Radeon 8060S 単体（HIP ビルド。エキスパートの一部は SSD から） | – | 7K の prompt 約 150 tok/s |

遅い 2 枚目を足しても、今のところ RTX 5090 1 枚 + iGPU レーンには届かない。カード間の受け渡しと遅いリンクのコストが、
RAM 層から減るエキスパートの分を上回る。分割時はまだ投機デコードも使えない。

</details>

## 🚀 クイックスタート

```sh
git clone https://github.com/zephel01/Abyss.git ~/llm/apps/Abyss
cd ~/llm/apps/Abyss
./setup-deepseek.sh --download-model     # エンジン、モデル（96.8 GB）、pack、設定、../Abyss-tools
../Abyss-tools/abyss.sh start            # → http://127.0.0.1:8082/
```

API で試す:

```sh
curl http://127.0.0.1:8082/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"日本の首都はどこですか？"}],"max_tokens":4000}'
```

<details>
<summary><b>セットアップがやること</b></summary>

1. PC の確認（NVIDIA GPU、RAM）
2. Python パッケージを `.venv` に入れ、固定コミットの llama.cpp のソースを取る
3. GPU に合わせてエンジンをビルド（RTX 50 は `120a`）
4. モデルを `../Abyss-data` にダウンロード（`--download-model` がなければ先に確認）し、全ファイルの SHA-256 を照合
5. シャードと同じフォルダに pack を作る
6. `abyss-<名前>.json` と `run-abyss-<名前>.sh` を書く
7. `../Abyss-tools` を作る — `abyss.sh start | stop | restart | status | update | build | log | models`

済んだ手順は飛ばすので、何度実行してもよい。オプション（`--quant`、`--context`、`--port`、`--gpus 0,1`（層の分割）、
`--backend hip`（AMD）、`--nvcc`（V100 用の CUDA 12）、`--tools-only` など）は
[マニュアル](docs/MANUAL.ja.md#まとめてセットアップ)。

</details>

### 必要なもの

| | |
| --- | --- |
| OS | Linux（Ubuntu で確認） |
| GPU | NVIDIA: RTX 5090 32 GB と RTX 3090 で測定。V100 は CUDA 12 でビルドできる（実機は未確認）。AMD Radeon は HIP で試験的に対応: Radeon 8060S で動作、単体カードは未確認 |
| メモリ | 全エキスパートをメモリに置くには VRAM + RAM が約 100 GB 以上（少なくても SSD から読んで動く） |
| ディスク | 空き約 100 GB の NVMe SSD |
| ツール | CUDA 13.x（`nvcc`。V100 は 12.9）、g++、Python 3.10 以上。AMD APU の iGPU レーンと AMD ビルドには ROCm 7（`hipcc`） |

## 🧠 しくみ

```mermaid
flowchart LR
    T[トークン] --> A["アテンション<br/>ウィンドウ + 圧縮 KV<br/>インデクサ top-512"]
    A --> R{"ルーター<br/>sqrt(softplus)"}
    R --> V["VRAM 層<br/>よく使うエキスパート"]
    R --> P["pinned RAM 層<br/>iGPU / CPU で計算、または PCIe で転送"]
    R --> D[("NVMe SSD<br/>あまり使わないエキスパート")]
    V --> O[次の層]
    P --> O
    D --> O
```

各層のアテンションは GPU で計算し、ルーターが 256 個から 6 個のエキスパートを選ぶ。VRAM にあるものはそのまま計算する。
RAM 層のものはその場で計算する（APU の内蔵 GPU があればそれが、なければ CPU のスレッドが）か、PCIe で GPU に送る
（配分は起動時に測って決める）。残りは SSD から読み、使われるほど上の層に上がる。長い prompt はエキスパートの行列積
（MMQ）と tensor core の attention を使うまとめ処理で読む。応答が文中の並びを繰り返すときは、n-gram の候補を次の
トークンと一緒に確かめる。
詳しくは [docs/deepseek4/PORT.md](docs/deepseek4/PORT.md)、計算の仕様は [docs/deepseek4/SPEC.md](docs/deepseek4/SPEC.md)。

## ✅ 正しさの確認

同じ GGUF で、llama.cpp の logits と比べた（`tools/dsv4_logits_dump.cpp`）:

| テスト | 結果 |
| --- | --- |
| 3,000 トークンの文書 | KL 0.009 · top-1 一致 94.5% · 次トークン的中率 60.35%（llama.cpp も 60.35%）· 位置が進んでもずれは増えない |
| まとめ処理で 2,499 トークン → 500 トークン | KL 0.006 · top-1 一致 95.2% |
| トークナイザー（`joyai-llm`） | `llama-tokenize` と同じ ID |

速度の変更はすべて同じ方法で確かめ直している（リリースごとの KL は[更新履歴](CHANGELOG.md)）。tensor core のカーネルは
FP32 のものと 2e-6 程度で一致し、投機デコードは 1 トークンずつの処理と同じ logits を返す。

## 📚 ドキュメント

| | |
| --- | --- |
| [マニュアル](docs/MANUAL.ja.md) · [Manual](docs/MANUAL.md) | セットアップ、設定、API、複数 GPU、AMD、設定項目、トラブル対応 |
| [更新履歴](CHANGELOG.md) | 各リリースの内容と測定結果 |
| [移植の記録](docs/deepseek4/PORT.md) | Abyss が足したものと、その確認方法 |
| [仕様](docs/deepseek4/SPEC.md) | 実装した DeepSeek V4 の計算 |
| [Maya のドキュメント](docs/maya/README.md) | Maya の文書をそのまま残したもの: GLM-5.3-Flash、インストーラ（`./abyss.sh`）、チューニング、AMD、複数 GPU |

## 🗺️ ロードマップ

- [x] DeepSeek V4 Flash: 1 トークンずつの処理、まとめ処理、会話の再利用
- [x] 思考レベル、DSML のツール呼び出し、OpenAI / Anthropic 互換 API
- [x] コマンド 1 つのセットアップと `Abyss-tools`
- [x] iGPU レーン（Ryzen AI Max）。生成と prompt の両方
- [x] 文章自体から候補を作る投機デコード（n-gram）。既定でオン
- [x] 長いコンテキスト: indexer と prompt の attention を tensor core で（125K まで確認）
- [x] NVIDIA の複数 GPU（層の分割。RTX 5090 + RTX 3090 で測定）
- [x] AMD（HIP、試験的）: Radeon 8060S で WMMA カーネルを使って動作
- [x] V100（sm_70）を CUDA 12 でビルド
- [ ] DSpark のドラフトモデル（この GGUF には MTP ブロックがない）
- [ ] 会話のディスク保存
- [ ] DeepSeek V4 での Windows（確認したのは Linux）
- [ ] 複数カードでの速度: 分割時の投機デコードと prestage、V100 の実機、単体の Radeon
- [ ] 短い prompt のまとめ処理の調整（8K では RAM 層のエキスパートを GPU に送る時間が大半）

## 🙏 クレジット

Abyss は **[Project Maya](https://github.com/mw00/project-maya)**（MIT、mw00 とコントリビューター）をコピーして作った
フォークで、Maya は **[Strata](https://github.com/Niko1221/Strata)**（MIT、Niko1221 と Strata のコントリビューター）から
生まれた。Maya の GLM-5.3-Flash の実行部、エキスパートの配置、まとめ処理、サーバー、ダッシュボードはそのまま使い、そこに
DeepSeek V4（llama.cpp の `deepseek4` アーキテクチャ）を GLM-5.3-Flash と並べて足した。Maya の README と文書は
[docs/maya/](docs/maya/README.md) に残している。DeepSeek V4 の計算は [llama.cpp](https://github.com/ggml-org/llama.cpp) の
実装に、量子化モデルは [unsloth](https://huggingface.co/unsloth) のものに拠っている。すべてに感謝する。

## 📄 ライセンス

Maya・Strata と同じく MIT — [LICENSE](LICENSE)。モデルの重みはそれぞれのライセンスに従う（DeepSeek V4: MIT）。
