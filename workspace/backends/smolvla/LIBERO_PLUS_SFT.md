# LIBERO-PLUS版からの8タスクSFT（2026-11-01）

初期モデルは取得済み `lerobot/smolvla_libero_plus@7bb70aa5bc92b82c9239142775d3a173103567ff`。
初期重みSHA256は `a67a056ab0761afa5f5efef8edd45065cef4494735231b7eeeb8328e71fcfb72`。
汎用モデルからの前回runとは出力を分離する。

- 原案8タスク全3,071episode、episode単位の既存train/validation分割を再使用。
- 同時刻のfront/wrist画像・8D state・7D relative OSC action、20Hz、chunk50。
- 2視点の順番はfront→wrist。公開モデルの観測feature定義にある6D state・5cameraの古い記載を、実データの8D・2cameraへ合わせる。内部の32次元paddingと重みの形は維持する。
- 初期モデル付属のnormalizerのmean/stdを使用する。実際の保存統計がstate8D/action7Dであることを検査した。新たな統計の再推定による入力スケール変更は行わない。
- strict読み込みに成功し、state projection、action input/output projectionの値が初期checkpointと完全一致することをGPU上で確認。記録は `models/libero-plus-7bb70aa5/initialization-audit.json`。
- batch4、accumulation1、Action Expertとstate projectionを追加SFT。vision/VLM固定。
- 10,000steps、LR 1e-5、warmup1,000steps、checkpoint1,000stepsごと。検証は8タスク各8window、計64windowを500stepごとに評価する。
- 設定・train loss・validation loss・LR・epoch相当値・stepはW&B offlineとmetrics.jsonlに記録。依存とlockは従来どおり保持。
- 正常終了後に最終checkpointで8タスク×10回、計80回評価。結果・全試行動画を保存する。

学習元のLIBERO-PLUSデータが重なる可能性があるため、validation lossを未見タスクへの汎化性能とは扱わない。成功率70%の判定は環境内の80試行で行う。

## ジョブ

サービス名: `physicalai-smolvla-libero-sft-eval80.service`

保存先: `/home/lemonydrink/PhysicalAI/xvla-libero-plus-ros2/backends/smolvla/runs/libero-plus-sft-eval80-20261101/`

`status.json`で進行段階、`job.log`で実行ログ、`train-*/metrics.jsonl`で実際の更新stepを確認できる。
学習済みモデルは`train-*/ckpt-010000/`、評価は`evaluation80/summary.json`へ保存する。
会話を閉じても継続する。PCとユーザーサービスが動いている必要がある。

```bash
systemctl --user status physicalai-smolvla-libero-sft-eval80.service
```

手動で別runを実行する場合（登録済みジョブと同時実行しない）：

```bash
cd /home/lemonydrink/PhysicalAI/xvla-libero-plus-ros2
wandb login
export SMOLVLA_CONFIG="$PWD/backends/smolvla/configs/libero-plus-sft.json"
WANDB_PROJECT=physicalai-smolvla-eight WANDB_MODE=offline \
  backends/smolvla/.venv/bin/python backends/smolvla/run.py train \
  --manifest backends/smolvla/data/full-v1/manifest.json \
  --output backends/smolvla/runs/libero-plus-manual
```

`WANDB_ENTITY`でentity、`WANDB_MODE`または`--wandb-mode online|offline|disabled`で記録モードを指定できる。登録済みジョブはoffline固定。
