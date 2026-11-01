# batch32へ切り替え

実験計画の変更によりphysical batchを4→32、gradient accumulationを8→1へ変更。
有効batch32、LR/scheduler、7D loss、FP32 trainable weights/BF16 autocast、weight decay、clip、入力規約、1epochの終端は維持する。

- 新設定: `configs/staged-batch32.json`
- 新ジョブ: `runs/staged-batch32-run/`
- サービス: `physicalai-smolvla-batch32.service`
- 再開元: `runs/staged-one-epoch-run/train-001/ckpt-002000`
- 重み、AdamW状態、Torch RNG、学習順序の消化位置64,000/364,505を継承。
- 未保存だった約400更新分は再実行する。最終学習軌跡における1巡の消化件数は変わらない。
- baseline80と2,000step評価は完了済みの結果を参照し、再評価せず学習から直接続行する。
- baseline結果と2,000step評価は元ジョブへのsymlinkで保持する。
- 最終step11,391まで継続し、4,000/6,000/8,000/10,000/11,391でcheckpointとrolloutを行う。
- checkpoint選択には既存2,000stepモデルも含める。
- 実batchを変更したため、乱数生成のshapeや数値演算順は異なり、microbatch版とbitwise同じ更新にはならない。

再開時にconfig変更を無条件には許可しない。今回許可するのはbatch_size/gradient_accumulation_stepsの2項目だけで、積32と他の学習設定・データ・サンプル順序が一致することを検査する。

rollout前は未使用のCUDA cacheを開放し、学習モデルと評価モデルが同居できる余裕を確保する。これは学習batchを下げる処理ではない。
validationのoffline loss計算は従来のbatch4を維持する。学習の全更新はbatch32、端数は損失マスクで処理する。

進捗: `status.json`、`job.log`、`train-*/metrics.jsonl`。
metricsの `batch/microbatches` が1、通常更新の `batch/real_samples` が32となることを確認する。
W&B offlineへ新設定、batch migration、loss、validation、LR、epoch/step、rolloutを記録する。

```bash
systemctl --user status physicalai-smolvla-batch32.service
```

手動の場合は通常の `wandb login` と `WANDB_PROJECT` / `WANDB_ENTITY` / `WANDB_MODE`、または `--wandb-mode online|offline|disabled` を利用できる。登録ジョブはoffline固定。APIキーをコードに保存しない。
