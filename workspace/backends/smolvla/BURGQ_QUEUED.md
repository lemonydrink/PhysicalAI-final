# BurgQモデルの次実験予約（2026-11-01）

**その後のユーザー指示により即時開始へ変更。** 先行のbatch32学習はユーザー指定で停止し、BurgQはGPU監査を通過して学習前80回評価を実行中。以下の正常終了待ちは、当初の予約条件の記録。開始条件の上書きはplan.jsonに保存した。

最新の入力・分割は [データ整合監査](BURGQ_DATA_AUDIT.md) を参照。学習365,887フレーム、11,434更新、warmup572。旧分割の11,391更新計画を置き換えた。

対象: https://huggingface.co/BurgQ/smolvla-libero-bs32-100k

## 予約条件と順序

現在の `staged-batch32-20261101` をそのまま継続する。
この先行ジョブのstatusがcompleteとなり、サービス `physicalai-smolvla-batch32.service` もinactiveになるまで、新実験は待機する。
単なるsystemd After指定ではなく、この両条件を30秒間隔で確認する。失敗・停止中の先行ジョブを正常完了と見なして追い越さない。

その後は以下を自動実行する。

1. BurgQ重みの厳密なGPU読み込み、正規化の往復、7次元lossの勾配、入力規約を監査する。この監査ではoptimizer更新しない。
2. **BurgQモデル自身の追加学習前baselineを、原案8タスク各10回、合計80回評価**する。既存モデルのbaselineは流用しない。
3. BurgQ公開重みから、新しいoptimizerで1epoch SFTを開始する。現在のモデルの重み・optimizer状態は継承しない。
4. 2,000optimizer stepsごと＋最後に80回rolloutし、タスク別成功率とmacro成功率を記録してbest checkpointを選ぶ。

## 固定モデル・互換性

- repo: `BurgQ/smolvla-libero-bs32-100k`
- revision: `238fb0654cd598e287f8b6c2d64bd09af239c76f`
- 重み: 1,197,789,224 bytes
- SHA256: `88638a6aa1399327d79418265b224b883e7af6b5a7855c3b0aa6b7929ff1f971`
- 取得先: `models/burgq-libero-100k-238fb065/`
- 使用先: 同ディレクトリの`compatible/`

公開元の重みSHA256と一致し、CPU上のstrict読み込みで500テンソルすべての値が一致した。
モデル付属のstate8D/action7D mean/stdを使用する。LIBERO-PLUS版のnormalizerは継承しない。

公開元configの`pretrained_revision: null`だけが現在のLeRobot0.4.4で未対応のmetadata項目だったため、compatible/config.jsonでこのnull項目だけを省いた。元configを保持し、変更内容とハッシュをprovenance.jsonに記録した。重み・構造・normalizer・processorには変更を加えない。
`load_vlm_weights=False`を維持し、このcheckpointに含まれる完全なVLM重みをロードする。sourceがFP32のfrozen weightsを、別の初期VLM読み込み経路でBF16へ丸めない。

公開元はLeRobot0.6.1 / Transformers5.5.4での評価を説明している。こちらでは現行LeRobot0.4.4 / Transformers4.57.6の厳密な重み互換性を確認し、実行前のGPU監査も行う。
これは公開元の40task・83%という数値の再現実験ではなく、現在の8task・固定初期状態での新しいbaselineとSFT比較である。

## 学習・評価条件

直近のbatch32設定を維持する。

- 学習: 原案8task、train2,765episode / 364,505frame、validation306episode / 40,211frame。
- batch32、勾配蓄積1、effective batch32。
- 1epoch=11,391optimizer updates。最大20,000stepより前に終了。
- Action Expertとstate/action projection中心。VLM/vision固定。
- trainable FP32、BF16 autocast、AdamW、LR3e-5、warmup570、cosineで最終3e-6、weight_decay1e-4、clip norm1.0。
- 有効7D/time mask付きloss、episode末尾も含め全train観測を各1回。
- dataset画像は無回転、simulator両cameraを180度回転してcanonical画像へ。8D stateは生成元と同じrobosuite quat2axisangle規約。
- baseline/rollout: 各task10回、初期state index10〜19、最大300step、20Hz。新モデル専用の結果・動画を保存。
- checkpoint選択: 成功総数、同点なら最低task成功率、成功task数、早いstep。同じ80条件を繰り返すため、selection用の評価として扱う。

## ジョブと記録

予約サービス: `physicalai-smolvla-burgq-queued.service`

出力:
`/home/lemonydrink/PhysicalAI/xvla-libero-plus-ros2/backends/smolvla/runs/burgq-staged-batch32-20261101/`

設定: `configs/burgq-staged-batch32.json`

- plan.json: 先行ジョブ、固定revision、開始条件、全設定のhash
- status.json: 待機・preflight・baseline・training・評価・complete/failed
- job.log: 実行ログ
- audit/: 引き継いだ入力監査と、新モデルのGPU監査
- baseline80/: 新モデルの学習前80件
- train-*/: 学習設定・metrics・W&B offline・checkpoint・coverage
- eval-*/: checkpoint別80件
- best_checkpoint.json: 最良checkpointとbaseline比較

```bash
systemctl --user status physicalai-smolvla-burgq-queued.service
```

会話を閉じても待機・実行を続ける。PCとユーザーサービスが稼働している必要がある。
待機中にactive_backendの現行モデルを変更しない。先行ジョブが終了して新モデルのpreflightを通過した時点で切り替える。

W&Bはoffline、project physicalai-smolvla-eight、新しいjob group/outputへ保存する。APIキーはコードに保持せず、通常の `wandb login`、WANDB_PROJECT / WANDB_ENTITY / WANDB_MODE、学習CLIの--wandb-mode online|offline|disabledが利用可能。登録済みジョブはoffline固定。
