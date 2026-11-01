# 4台並列評価への切替

記録：2026-11-01T23:06:55.019246+09:00。元レポートは22:31時点の中間snapshotとして保持する。

ユーザー指示により2,000更新checkpointの評価からシミュレーター4台へ切り替えた。完了済み18試行を保持し、未完了分を4workerへ分配。保存済みcheckpointからoptimizer・データ順・乱数状態を復元している。学習recipe/configとtrain/eval splitは変更していない。

## 実装

- 4個の独立したsimulator processを18041〜18044番portへ起動。各workerに別のLIBERO_CONFIG_PATH用ディレクトリを用意する。
- モデルは1個を共有。ThreadPoolExecutorの4workerがHTTPのreset/stepと動画保存を並列に処理する。
- GPU推論はlockして直列実行する。4モデルの同時推論や推論batch4を導入したわけではない。
- 各trialのaction queue、CPU/CUDA RNGを独立保存し、推論前後に切り替える。seed・初期状態・評価条件は従来どおり。
- 同じprotocolとモデルhashに合う完了済みtrialだけ再利用し、同じtrialを複数workerで実行しない。
- 評価が全80件揃ったらW&Bへ記録し、学習を再開する。以後のcheckpoint評価も4台構成を使う。
- 実行metadataは評価出力のexecution.jsonへ保存する。W&B評価runにもworker数と実行時間を記録する。

## 確認

backendテスト14件、coreテスト15件が成功。新規テストでは4threadを交互実行し、trialごとの乱数とaction queueが逐次実行と一致すること、呼び出し元の学習RNGが変更されないことを検査した。
4個のsimulator healthがすべて正常で、4workerが実際にtrialを完了したことを確認。稼働中の観測ではGPU27%、VRAM6,239MiB、ホスト利用可能RAM約2.9GiB。値は瞬間値であり平均ではない。

速度倍率はまだ確定していない。逐次時と残試行の構成が異なるため、単純な試行数/分の比を厳密な性能比較とは扱わない。今後のexecution.jsonを使って評価全体の時間を確認できる。

公開時点ではこのcheckpointの80件評価は進行中。SFTの最終結果はまだ未確定である。
