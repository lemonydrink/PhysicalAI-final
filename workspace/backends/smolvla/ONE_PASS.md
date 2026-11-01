# 学習データ全件を1巡して検証（2026-11-01）

直近のLIBERO-PLUS版SFT checkpoint `runs/libero-plus-sft-eval80-20261101/train-001/ckpt-010000` から継続する。

- 学習: 2,765episode、364,505フレーム。すべての観測時刻を1回ずつシャッフル順で使用し、ランダムな復元抽出は行わない。
- 以前の229,020個は「50stepを末尾まで取れる開始点」の数だった。今回の全件処理では末尾49step以内も含める。エピソード終端以降のactionはpaddingし、損失から除外する。別episodeのactionを跨いで使わない。
- batch4。追加91,127回更新し、累積stepは10000→101127。最後のbatchは実データ1件と損失ゼロのダミー3件。学習に寄与する観測は各1回。
- LR1e-5、既存optimizerと重みを継続。元モデルのnormalizerとAction Expert/state projectionの学習対象を維持。
- 5,000stepごとにcheckpointと8タスクの小規模検証を記録する。
- 最後に検証用306episode・40,211フレームすべてで損失を計算する。検証用データで重みを更新しない。
- その後、8タスク×10回の80回シミュレーション評価を実行し、動画・成功率を保存する。

学習順は`epoch_order.pt`とSHA256で記録する。checkpointに何件消化したかを保存し、再開時には同じ順番の続きから処理する。終了時の`coverage.json`に学習全件と検証全件の件数を保存する。
マスク入力は、この固定版LeRobotの`actions_id_pad`というキーへ明示的に渡している。実モデルで全paddingの損失がゼロになることをGPU上で確認した。

## 実行状況

ジョブ: `physicalai-smolvla-one-pass.service`

出力: `/home/lemonydrink/PhysicalAI/xvla-libero-plus-ros2/backends/smolvla/runs/libero-plus-one-pass-20261101/`

`status.json`、`job.log`、`train-*/metrics.jsonl`で進捗を確認する。
`epoch`は今回の追加1巡の進捗で、完了時に1となる。`coverage/consumed_frames`の完了値は364505。
最終重みは`train-*/ckpt-101127/`、成功率は`evaluation80/summary.json`。
W&B offlineへ設定、train loss、validation loss、LR、step、epoch、消化フレーム数を記録する。

```bash
systemctl --user status physicalai-smolvla-one-pass.service
```

会話を閉じても継続する。PCとユーザーサービスの稼働が必要。

## 手動実行例（登録済みジョブと同時に実行しない）

```bash
cd /home/lemonydrink/PhysicalAI/xvla-libero-plus-ros2
wandb login
export SMOLVLA_CONFIG="$PWD/backends/smolvla/configs/libero-plus-one-pass.json"
WANDB_PROJECT=physicalai-smolvla-eight WANDB_MODE=offline \
  backends/smolvla/.venv/bin/python backends/smolvla/run.py train-epoch \
  --manifest backends/smolvla/data/full-v1/manifest.json \
  --resume backends/smolvla/runs/libero-plus-sft-eval80-20261101/train-001/ckpt-010000 \
  --output backends/smolvla/runs/one-pass-manual
```

`WANDB_ENTITY`、`WANDB_PROJECT`、`WANDB_MODE`または`--wandb-mode online|offline|disabled`で設定できる。APIキーをコードに保持しない。
