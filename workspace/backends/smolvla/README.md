現行BurgQ実験の評価はシミュレーター4台＋共有モデルへ切替済み。plan.jsonのevaluation_workers=4。trial別RNG/action queueを分離し、評価のexecution.jsonとW&Bに実行情報を保存する。

# SmolVLA：8task追加学習

現行：BurgQの学習前80回評価→batch32・1epoch SFT。[データ整合監査](BURGQ_DATA_AUDIT.md)で重複軌道による検証漏洩を修正済み。学習365,887フレーム、11,434更新。以下の待機・旧主系統の記述は過去の実験経緯。

当初の実験予約：[BurgQモデルの学習前80回評価→batch32・1epoch SFT](BURGQ_QUEUED.md)。現在のbatch32ジョブが正常終了し、プロセスも退出するまで待機する。

現在の実行は[batch32・勾配蓄積1での継続学習](BATCH32.md)。完了済みbaselineを再利用し、保存済み2,000stepから直接再開する。

現在の主系統は[段階的SFT＋rollout選択・1epoch](STAGED_ONE_EPOCH.md)。公開モデルbaseline80回の後、batch4×蓄積8、7D loss、warmup/cosineで学習する。以下の以前の1巡設定・結果は過去実験の記録。

現在は[全データ1巡の追加学習・全件検証](ONE_PASS.md)を実行中。`configs/libero-plus-one-pass.json`と`train-epoch`を使用する。

2026-11-01の主系統は **LIBERO-PLUS学習済みモデルからのSFT**。設定は`configs/libero-plus-sft.json`、詳細は[現行SFT](LIBERO_PLUS_SFT.md)。`SMOLVLA_CONFIG`で設定を指定する。以下の汎用baseの記述・旧ジョブは前回実験の記録。

主系統の切替先。X-VLAの既存結果を学習成果に数えず、汎用`lerobot/smolvla_base`から原案8taskを追加学習して評価する。

- batch4、単一GPU、Action Expert/state projection SFT、vision/VLM固定。
- state8D、front/wrist、7D相対OSC action。絶対pose変換なし。
- train episodeだけからstate/action統計を計算する。
- W&B online/offline/disabled、設定・loss・validation・LR・epoch相当値・step・checkpoint対応を記録。
- 独立した`uv.lock`とローカルコピーのLeRobot source/hashを保持。旧環境へ追加インストールしない。

```bash
cd /home/lemonydrink/PhysicalAI/xvla-libero-plus-ros2
uv sync --locked --project backends/smolvla
backends/smolvla/.venv/bin/python backends/smolvla/run.py download
backends/smolvla/.venv/bin/python backends/smolvla/run.py prepare \
  --source data/smoke-split/manifest.json --output backends/smolvla/data/new-smoke
backends/smolvla/.venv/bin/python backends/smolvla/run.py train \
  --manifest backends/smolvla/data/new-smoke/manifest.json \
  --output backends/smolvla/runs/new-smoke --steps 2 --smoke --wandb-mode offline
```

取得済みraw動画とstate/actionを同時刻で再読込し、X-VLA用の生成target/先頭画像除去は利用しない。`--source`はepisode選択・splitの参照のみ。

```bash
wandb login
WANDB_PROJECT=physicalai-smolvla-eight WANDB_ENTITY=<entity> WANDB_MODE=offline \
  backends/smolvla/.venv/bin/python backends/smolvla/run.py train \
  --manifest backends/smolvla/data/full/manifest.json --output backends/smolvla/runs/sft-v1
```

full学習にはpartial=falseのfull datasetが必要。少数episodeのsmokeで性能を判断しない。出力先の上書きは禁止。checkpointにはpolicy/pre/postprocessor、optimizer、RNG、stepを保存する。`--resume <checkpoint>`で新しいoutputへ再開する。SmolVLAの同期rolloutは別Pythonのsimサービスを使用し、ROS adapterは切替中。X-VLAのROS経路をSmolVLA対応済みとは扱わない。

参考：[SmolVLA公式モデルカード](https://huggingface.co/lerobot/smolvla_base)。checkpointは`configs/train.json`のrevisionへ固定。

## 8タスクの元データ取得

全件取得手順・件数・検証方法は[データ取得記録](../../docs/DATASET_ACQUISITION.md)を参照。取得完了記録は`data/eight-tasks-local.manifest.json`。上記`smoke`は少量の動作確認データであり、全件データとは区別する。

## 登録済みの本学習・評価ジョブ

[予約内容と実行状況](QUEUED_JOB.md)：全件データ準備 → batch4で10,000step SFT → 8タスク×10回評価。ジョブを起動済み。完了・70%達成の判定は`status.json`と評価結果で確認する。

## 現在の検証結果

batch4、2 optimizer stepsの実GPU試験が完了。validation4batch、checkpoint保存、W&B offlineログを確認。peak VRAMは約2.04 GiB。これは疎通確認であり成功率評価ではない。`tests/test_contract.py`の2試験と依存整合検査も合格。

Torch2.10/cu130は最小BF16行列演算でもCUBLAS初期化エラーとなったため、動作確認済み2.11/cu130へ変更。コピー内LeRobot依存上限だけを`compat/torch-2.11-bounds.patch`で緩和し、source hashとlockへ記録した。

## 同期シミュレーション評価

旧NumPy1環境のsim workerを起動する（既存PARCを変更しない）。

```bash
.venv/bin/vla serve --kind sim --port 8011
# 別ターミナル：未調整初期点の確認
backends/smolvla/.venv/bin/python backends/smolvla/run.py evaluate \
  --manifest backends/smolvla/data/smoke/manifest.json \
  --output backends/smolvla/runs/base-check --task milk_basket --episodes 1
# 学習後は同じmanifest・task・初期stateで --checkpoint <保存先> を追加してpaired比較する。
```

汎用初期点の6D state/6D action定義を8D state/7D actionへ合わせる。内部paddingは32Dであり、7D成分が学習前からLIBEROの意味に適応済みとは仮定しない。学習後の改善と目標70%はrolloutで判定する。
