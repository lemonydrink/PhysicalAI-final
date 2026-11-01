現行実験は完了。[モデル比較と最終ROS 2評価を分けた最終報告](../reports/experiment-burgq/final-report/REPORT.md)を参照。HTTPモデル比較91.25%→97.5%、最終ROS 2評価97.5%（別初期状態）。以下は過去の作業履歴です。

# 公開スナップショットについて

現行BurgQ実験の学習前baselineは73/80（91.25%）、追加学習はbatch32で進行中です。[最新レポート](../reports/experiment-burgq/REPORT.md)を参照してください。以下は過去の作業履歴を含む元READMEです。

# 現在の主系統：SmolVLA

実験条件として、**汎用SmolVLAから原案8taskを追加学習する構成へ切替中**です。batch4を維持します。[SmolVLA実装・実行手順](backends/smolvla/README.md)と`configs/active_backend.json`を参照してください。SmolVLAの70%目標は未達・未評価です。

以下はX-VLAの既存参照実装と評価履歴であり、その97.5%を今回の学習成果には数えません。

# SmolVLA × LIBERO-PLUS × ROS 2（X-VLA比較履歴あり）

2toINF upstream版X-VLAを原案の8タスクに接続する、シミュレーション専用プロジェクトです。目標は8タスクmacro成功率70%以上と、全8タスクの成功動画。実機は使用しません。学習batch sizeは4です。

**確認評価：78/80成功（97.5%）、衝突なし58/80（72.5%）。** 追加学習前のベースモデルで達成しました。[全8タスクの成功動画と検証結果](docs/IMPLEMENTATION_STATUS.md)を参照してください。

## 導入と資産

```bash
cd ${WORKSPACE}
bash scripts/bootstrap.sh
```

`uv.lock`でML/sim依存を固定します。既存PARCのvenv・コード・重みを変更しません。ROSは`../ros2`の独立したPixi環境を使用します。LIBERO-plusは`third_party/libero`へコピーしたコードと既存の固定assetsを参照します。出所・hashは`docs/UPSTREAM_SOURCE.json`と`docs/LIBERO_SOURCE.json`を参照してください。assetsシンボリックリンク先を移す場合は同じrevisionの資産を再配置してください。

ベース重みは`models/base/model.safetensors`へ固定revisionから取得し、予選で記録されたSHA-256を照合します。外部ソースのダウンロードコードを動的実行する構成ではありません。

## シミュレーション・動画・成功率

```bash
.venv/bin/vla doctor
.venv/bin/vla evaluate --output runs/my-baseline --frame controller --per-task 10 --offset 10
.venv/bin/vla score runs/my-baseline --per-task 10
```

`data/tasks.resolved.json`が公開datasetのtask index・指示と基本BDDL/init-stateを結び付けます。対象は牛乳、トマトソース、バター、ボウル3種、引き出し、皿押し。初期環境は摂動なしの基本8taskであり、全LIBERO-PLUS摂動の評価ではありません。

各trialのJSONと連続MP4を保存します。モデル/実行profile/seed/初期state hash/衝突・timeout・infra errorを分けて記録し、初期stateを循環利用しません。開発smokeはoffset0、確認評価はoffset10を使用します。`score`は80trialすべてと全taskの成功動画を要求し、macro率を計算します。成功率の主指標はtask completion、PARC式非対象物1mm変位によるcollision-free率は補助指標です。

`--frame body`は予選body-frame経路の比較用。Baseではcontroller-frame経路を開発評価で選択しました。学習データのbody-frame Codecとは分けて管理し、未検証の変換を本学習に使用しません。

```bash
.venv/bin/vla export --output models/export-base-v1
.venv/bin/vla verify-bundle models/export-base-v1
```

exportには独立したbase重み、processor、ソース、lock、設定、task対応を含みます。上書きせず、3.5 GB前後の保存容量を使います。LoRA使用時は`--adapter runs/<run>/ckpt-XXXXXX`を指定します。モデルbundleの`PYTHONPATH=src`で`physicalai_vla.model.Policy`をロードできます。シミュレーション実行には別途LIBERO/scene assetsが必要です。

## データ準備とbatch4学習

```bash
# 少数データ：各task2episode。性能判断用の学習ではありません。
.venv/bin/vla prepare --max-per-task 2 --output-dir data/smoke-v1
.venv/bin/vla train --manifest data/smoke-v1/manifest.json \
  --output runs/train-smoke --steps 2 --smoke --wandb-mode offline

# フルデータ。時間・容量を要します。
.venv/bin/vla prepare --output-dir data/full-v1
.venv/bin/vla train --manifest data/full-v1/manifest.json \
  --output runs/train-full --wandb-mode online
```

LoRA r8/alpha16/all-linear＋Soft Prompt＋Action Encoder/Decoder、batch4、AdamW、BF16を実装しています。validation loss、train loss、LR、epoch相当値、step、checkpoint対応をW&BとJSONLへ記録します。データはepisode単位でtask別分割し、末尾の不足windowを使用しません。raw20Hzの時系列を維持するので、過去handlerの30Hz再補間とは異なる条件です。

**フル学習はCodecの教師再生監査が完了し、`configs/codec.json`がvalidatedとなった場合のみ開始します。** 現在のCodecは未検証の候補で、state/actionの同時刻か1stepずれかを未確定のまま本学習へ進めません。`--smoke`は最大2追加更新だけ、このゲートを迂回します。損失やsmoke checkpointを採用モデルとは扱いません。

checkpointにはadapter、processor、optimizerとRNGを保存します。

```bash
.venv/bin/vla train --manifest data/smoke-v1/manifest.json \
  --resume runs/train-smoke/ckpt-000002 \
  --output runs/train-resume-smoke --steps 4 --smoke --wandb-mode disabled
```

```bash
wandb login
WANDB_PROJECT=physicalai-xvla-eight WANDB_ENTITY=<entity> WANDB_MODE=online \
  .venv/bin/vla train --manifest data/full-v1/manifest.json --output runs/new-run
# WANDB_MODE=offline / disabled、または --wandb-mode で切替。
```

APIキーはコード・設定へ保存しません。CUDA利用不可時はCPUへ黙って切り替えず停止します。config/lock/datasetのhashを実行記録へ保存します。

## ROS 2接続

Python3.12のML/simプロセスと、JazzyのROS PythonをHTTP IPCで分離します。ROS Pythonへtorchを導入する必要はありません。

```bash
bash scripts/build_ros.sh
```

ターミナル1/2（新プロジェクトのルート）：

```bash
.venv/bin/vla serve --kind policy --port 8010
# 別ターミナル
.venv/bin/vla serve --kind sim --port 8011
```

ターミナル3/4（既存venvをdeactivateしたbash）：

```bash
source ../ros2.env
source ros2_ws/install/setup.bash
python scripts/ros_nodes.py --role policy --url http://127.0.0.1:8010
# 別ターミナルでも同じsourceを実行
python scripts/ros_nodes.py --role bridge --url http://127.0.0.1:8011 --task milk_basket
```

`/vla/observation`、`/vla/action_chunk`、`/vla/status`を使用します。policy同時処理は1、待機は最新観測のみ。UUID/sequence/codec/有限値/shape/期限を検証し、無効chunk・buffer不足では最後の動きを繰り返さずsim時間進行を止めます。現在のROS profileは実測に合わせた観測age上限1秒で、実機安全保証を目的としません。

ROS経由はmeasured-proprio非同期profile、直接評価はprevious-chunk-proprio同期profileです。**直接評価の成功率をROS経由の成功率とは扱いません。** 現在のROS proxyは単一task接続用で、80trial動画評価は直接推論CLIを使用します。

## 検証と設計資料

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q src scripts
uv pip check --python .venv/bin/python
source ../ros2.env
source ros2_ws/install/setup.bash
python scripts/smoke_ros.py
```

- [要件定義書](docs/REQUIREMENTS.md)：最新のシミュレーション限定スコープを優先
- [PARC照合書](docs/PARC_CROSSCHECK.md)：過去の成功/失敗と適用限界
- [意思決定台帳](docs/DECISIONS.md)
- [原案](docs/source/technical_design_v1.0_original.txt)：改変せず保存
