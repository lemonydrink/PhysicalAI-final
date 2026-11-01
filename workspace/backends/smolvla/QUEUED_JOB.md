# SFT → 80回評価の予約（2026-10-30）

ローカルの常駐ジョブ `physicalai-smolvla-sft-eval80.service` に登録し、起動した。
PCが稼働しユーザーのサービスが動いている間、会話を閉じても実行を継続する。

1. 検証済み原案8タスク全3,071episodeから学習データを作成する。episode単位で約90% train / 10% validationに分割し、正規化統計はtrainだけから求める。元データの摂動系譜が不明な点は維持する。
2. 汎用SmolVLAからAction Expert/state projection SFTを行う。batch 4、10,000 steps、LR 1e-4、warmup 1,000、checkpoint 1,000 stepsごと。設定はconfigs/train.json。W&B offlineで設定、train loss、validation loss、LR、epoch相当値、stepを記録する。
3. 最終10,000step checkpointを、原案8タスク各10回、合計80回評価する。初期状態index 10〜19、1試行最大300step。全試行の動画・成功判定を保存する。
4. タスク別成功率、全体macro成功率、8タスクの成功動画の有無を集計し、W&B offlineへ評価値も記録する。目標70%は結果を見て判定する。

学習が正常終了した場合に限り評価を開始する。中断後は完成済み変換データ、最新checkpoint、完了済み評価記録を再利用する。異常終了はstatus.jsonとjob.logへ記録し、常駐ジョブは60秒後に再試行する（1時間内に3回まで）。

実行計画・状態・ログは以下のディレクトリに保存する。

`/home/lemonydrink/PhysicalAI/xvla-libero-plus-ros2/backends/smolvla/runs/sft-eval80-20261030/`

- `plan.json`: 登録した設定・データhash
- `status.json`: preparing / training / evaluating / complete / failed
- `job.log`: 各段階の実行ログ
- `train-*/metrics.jsonl`, `train-*/wandb/`: 学習指標とW&B offline記録
- `train-*/ckpt-010000/`: 評価対象の学習済み重み
- `evaluation80/summary.json`: 80回評価の集計
- `evaluation80/*.mp4`: 各試行の動画

```bash
systemctl --user status physicalai-smolvla-sft-eval80.service
# 中断する場合
systemctl --user stop physicalai-smolvla-sft-eval80.service
# 再開する場合
systemctl --user reset-failed physicalai-smolvla-sft-eval80.service
systemctl --user start physicalai-smolvla-sft-eval80.service
```

W&Bの通常の手動実行ではREADMEに記載の `wandb login` と `WANDB_PROJECT` / `WANDB_ENTITY` / `WANDB_MODE`、または `--wandb-mode online|offline|disabled` が利用できる。本予約はofflineに固定し、外部への自動アップロードを行わない。
