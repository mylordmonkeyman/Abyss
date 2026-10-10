# Abyss-tools

Abyss の起動・停止・更新をまとめた運用フォルダ。Abyss 本体の `setup-deepseek.sh` が作る（雛形は本体の
`ops/abyss-tools/`）。本体の repo とは別に置き、自分のメモや記録もここに書き足してよい（この README は
セットアップを再実行しても上書きされない。`abyss.sh` と `common.sh` は雛形が新しくなると置き換わり、元のものは
`.bak` に残る）。

```
~/llm/apps/
  Abyss/         本体（git 管理）。エンジン build/, 設定 abyss-<model>.json, 起動スクリプト run-abyss-<model>.sh
  Abyss-data/    モデル本体と pack（git の外。更新しても消えない）
  Abyss-tools/   ここ
```

Abyss は [Project Maya](https://github.com/mw00/project-maya)（MIT）をコピーして作ったフォークで、Maya は
[Strata](https://github.com/Niko1221/Strata)（MIT）から生まれたプロジェクト。マニュアルは本体の
`docs/MANUAL.ja.md`（英語 `docs/MANUAL.md`）。

## スクリプト

全部 `./abyss.sh <コマンド>` で使う（引数なしで一覧を表示）。本体の `abyss.sh`（Maya から引き継いだ GLM 用インストーラ）とは別物。

| | |
| --- | --- |
| `./abyss.sh start [model]` | ソースが変わっていればエンジンを再ビルドしてから `run-abyss-<model>.sh` をバックグラウンドで起動し、`ready` まで待つ。ログは `logs/server-<model>.log` |
| `./abyss.sh start -f [model]` | 同じだがこのターミナルで起動（Ctrl+C で停止）。systemd などの自動起動はこちら |
| `./abyss.sh stop` | 起動中の Abyss を止める（サーバーに SIGTERM = Ctrl+C と同じ後片付け。100 秒で止まらなければエンジンにも SIGTERM → SIGKILL） |
| `./abyss.sh restart [model]` | stop → start（モデル名を省くと起動中のモデル） |
| `./abyss.sh status` | git の位置と origin の未取得コミット数、エンジンがソースと一致しているか、起動中かを表示 |
| `./abyss.sh update` | `git pull --ff-only` → エンジンの再ビルド（起動はしない）。起動中・未コミットの変更ありなら止まる |
| `./abyss.sh build` | エンジンの再ビルドだけ（変更分のみ。停止中のみ） |
| `./abyss.sh log [model]` | サーバーのログを `tail -f` |
| `./abyss.sh models` | 使えるモデル（`run-abyss-*.sh`）とポートの一覧。`*` 起動中、`>` 最後に start したモデル |

`[model]` は `run-abyss-<model>.sh` の `<model>` 部分。省略時は、1 つしかなければそれ、複数あれば
起動中のモデル → 最後に start したモデル。別の場所の Abyss を扱うとき: `ABYSS_DIR=/path/to/Abyss ./abyss.sh status`

## 起動と停止

```sh
./abyss.sh start    # 起動（1 分ほど: 密な重みを GPU に載せ、エキスパートを VRAM と RAM に読み込む）
./abyss.sh log      # 様子を見る（Ctrl+C で抜けてもサーバーは止まらない）
./abyss.sh stop     # 停止
```

- ダッシュボード: `http://127.0.0.1:<port>/`、OpenAI 互換 API: `http://127.0.0.1:<port>/v1`、Anthropic 互換: `/v1/messages`
- ログは 2 つ: サーバー（`logs/server-<model>.log`）とエンジン（本体の `abyss-<model>.log`）
- 思考させるときは `max_tokens` を数千にする（サーバーは回答に最低 1024 トークンを残すので、数百だと思考がすぐ閉じられる）

## モデルを足す・設定を変える

本体の `setup-deepseek.sh` をもう一度実行する（済んだ手順は飛ばす）。別の量子化は `--quant`、コンテキストは
`--context`、ポートは `--port`。設定ファイル `abyss-<model>.json` を直接直してもよい（`"env"` にエンジンの設定）。

## 更新の手順

```sh
./abyss.sh stop
./abyss.sh update    # git pull + エンジンの再ビルド
./abyss.sh start
```
