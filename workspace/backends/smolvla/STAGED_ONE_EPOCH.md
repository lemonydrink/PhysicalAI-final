# 段階的SFT＋Rollout選択：1epoch

実験条件の新設定で公開モデルから学習し直す。以前の10,000stepや1巡追加学習の重み・optimizerは継承しない。

## 実行順序

1. 入力・normalizer・7D loss・FP32 trainable weightsを監査済み。
2. 修正した入力で公開 `lerobot/smolvla_libero_plus@7bb70aa5bc92b82c9239142775d3a173103567ff` を8タスク×10回、80回評価する。
3. 完了後、学習用364,505フレームを1epoch SFTする。
4. optimizer step 2000/4000/6000/8000/10000/11391で各80回rollout。
5. 最良の追加学習checkpointを選択。baselineとの比較も保存し、改善しなかった場合は明記する。

## 学習設定

| 項目 | 設定 |
|---|---|
| 初期モデル | 公開LIBERO-PLUS版、以前の追加SFTからは再開しない |
| 学習対象 | Action Expert、state/action projection、action/time MLP |
| VLM/vision | 凍結、eval modeを維持 |
| trainable weights | FP32保持、99,880,992 parameters |
| forward/backward | BF16 autocast、GradScalerなし |
| optimizer | AdamW、betas(0.9,0.95)、eps1e-8 |
| micro batch | 4 |
| gradient accumulation | 8 |
| effective batch | 32。最終更新だけ実データ25件（端数はマスク） |
| epoch | 全train観測364,505件を各1回、重複のないseed42 permutation |
| optimizer updates | ceil(364505/32)=11,391（上限20,000以内） |
| LR | 3e-5 |
| warmup | 今回の実更新数の5%＝570step、linear |
| decay | cosine、1epoch終了時3e-6 |
| weight decay | 1e-4 |
| gradient clipping | global norm1.0、蓄積後に1回 |
| loss | flow-matching velocity MSE、有効な7次元のみ |
| episode末尾 | 有効なtimeだけで平均、別episodeを跨がない |
| 最終端数 | batch4を維持しダミーの全timeをマスク、実サンプル数で平均 |
| chunk/n_action_steps | 50/10 |
| seed | 42 |
| checkpoint | 2,000optimizer stepsごと＋最終 |
| W&B | offline、project physicalai-smolvla-eight |

`scheduled_lr`はstep1から570まで線形増加し、以降step11391で3e-6に達する。20,000stepまで学習する設定ではなく、今回は1epochで終了する。

## 入力監査・修正

### 画像

同じカメラで実データとsimulator raw画像を比較した。両cameraで180度の向き差があるため、単純に既存回転を外す変更は行わない。

- dataset front/wrist: そのまま。
- simulator front/wrist: **両方を180度回転**してdataset側の向きへ揃える。
- RGB uint8、256×256→CHW/255。model内で512×512 bilinear resize/pad、[-1,1]へ。
- train/evalは共通の`input_contract.rgb_tensor`を使い、入力元に応じて同じcanonical画像へ変換する。
- 監査画像はジョブ配下audit/camera_contract.png（raw比較）、camera_canonical.png（向きを合わせた比較）。物体の初期配置は異なり、pixel単位で同一シーンという主張ではない。

### state

8D=[eef position xyz(m), LIBERO/robosuite quat2axisangleのaxis-angle xyz(rad), gripper qpos2(m)]。

以前はsimulatorでSciPy.as_rotvecを使っていた。rawデータ全404,716frame中106,735frameは姿勢ベクトルの長さがπ超であり、SciPyの主値表現では一致しない。
実reset時にも、データ生成元と同じ変換でx≒+3.1408、旧変換ではx≒-3.1397となるケースを確認した。

新profileはデータ生成のcreate_dataset.pyと同じ `robosuite.utils.transform_utils.quat2axisangle` を用いる。位置・gripper順序は同じraw観測から構成し、実simulatorと生成元の一致を確認した。

### action・正規化・loss

- 7D relative OSC、位置3・回転3・gripper1。20Hz。元のaction[t]をそのまま使用する。
- normalizerは公開初期checkpoint付属mean/stdを継承。state8D/action7Dの形状と正規化→逆変換の往復を検査済み。
- rawデータのaction範囲は監査JSONに保存。controller入力範囲[-1,1]内。
- 内部32D paddingは維持するが、raw flow-matching lossを[..., :7]で切り出し、有効time×7で平均する。
- 実モデルで追加25次元のoutput projection勾配がゼロ、実7次元は非ゼロであることを検査した。監査ではoptimizer更新を行っていない。
- 8D rawデータだけでは完全なsimulator状態を復元できないため、全デモの完全リプレイによる証明は行えていない。生成コード、数値分布、実観測との対応を監査根拠として記録する。

## 検証・選択

途中のoffline validationは8タスク各8点、計64点。最後はvalidation全40,211frameで7D/time mask付きの損失を計算する。

baselineとcheckpoint評価はすべて同じ80条件（8タスク各10、初期state index10〜19、seed=42+task_index*100000+episode_id、上限300step、20Hz）。

選択は成功総数（macro成功率）最大を優先し、同点なら最低タスク成功率、成功したタスク数、早いcheckpointの順で決める。タスク別成功率も必ず保存する。

同じ80条件を繰り返しcheckpoint選択に使うため、これはselection用のrolloutであり、独立した未使用testによる70%確認とは区別する。

## 状況・出力

サービス: `physicalai-smolvla-staged-one-epoch.service`

ジョブ: `${WORKSPACE}/backends/smolvla/runs/staged-one-epoch-run/`

- plan.json / status.json / job.log
- audit/contract.json、simulator_contract.json、監査画像
- baseline80/summary.json、全80試行のJSON・動画
- train-*/run.json、metrics.jsonl、wandb/、epoch_order.pt、coverage.json
- train-*/ckpt-002000/ … ckpt-011391/
- eval-002000/ … eval-011391/summary.json、試行JSON・動画
- best_checkpoint.json: 最良の学習checkpoint、baseline比、タスク別成功率
- source/: 設定・コード・lockとsimulatorソースのsnapshot

中断後は完成済みbaselineと最新checkpointを再利用する。checkpointのrollout未完了分を完了してから次の学習へ進む。
会話を閉じてもジョブを継続する。PCと実験実施者サービスが稼働している必要がある。

```bash
systemctl --user status physicalai-smolvla-staged-one-epoch.service
```

## W&Bと手動実行

```bash
cd ${WORKSPACE}
wandb login
export SMOLVLA_CONFIG="$PWD/backends/smolvla/configs/staged-one-epoch.json"
# baseline完了後に手動で起動する場合（登録済みジョブとは同時実行しない）
WANDB_PROJECT=physicalai-smolvla-eight WANDB_MODE=offline \
  backends/smolvla/.venv/bin/python backends/smolvla/staged_training.py \
  --manifest backends/smolvla/data/full-v1/manifest.json \
  --output backends/smolvla/runs/staged-manual \
  --job backends/smolvla/runs/staged-one-epoch-run
```

WANDB_ENTITY、WANDB_PROJECT、WANDB_MODEまたは--wandb-mode online|offline|disabledで変更可能。
登録済みジョブはoffline固定。設定、train loss、validation metrics、LR、epoch/step、grad norm、coverage、rollout task別成功率を記録する。
APIキーのコード保存・外部への自動アップロードは行わない。
