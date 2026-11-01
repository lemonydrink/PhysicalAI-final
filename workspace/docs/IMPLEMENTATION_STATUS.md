> この文書はX-VLAの参照評価履歴です。現在のSmolVLA追加学習の成果ではありません。主系統は`backends/smolvla`へ切替中です。

# 実装・評価結果（2026-10-30）

原案8タスクのシミュレーションで、2toINFベースモデル＋controller-frame同期実行が **78/80成功（97.5%）**。同じ全80trialに対するPARC式collision-free成功は **58/80（72.5%）**。8タスクすべてに成功動画があり、下表はcollision-freeな成功trialから1本ずつ選んだ。

| task | タスク完了 | 衝突なし完了 | 成功動画 |
|---|---:|---:|---|
| milk_basket | 9/10 | 9/10 | [動画](../runs/base-controller-confirmation80/milk_basket_005.mp4) |
| tomato_basket | 10/10 | 10/10 | [動画](../runs/base-controller-confirmation80/tomato_basket_003.mp4) |
| butter_basket | 10/10 | 10/10 | [動画](../runs/base-controller-confirmation80/butter_basket_003.mp4) |
| bowl_center_plate | 10/10 | 8/10 | [動画](../runs/base-controller-confirmation80/bowl_center_plate_006.mp4) |
| bowl_cookie_plate | 10/10 | 1/10 | [動画](../runs/base-controller-confirmation80/bowl_cookie_plate_005.mp4) |
| bowl_plate | 10/10 | 8/10 | [動画](../runs/base-controller-confirmation80/bowl_plate_000.mp4) |
| middle_drawer | 10/10 | 10/10 | [動画](../runs/base-controller-confirmation80/middle_drawer_005.mp4) |
| plate_stove | 9/10 | 2/10 | [動画](../runs/base-controller-confirmation80/plate_stove_007.mp4) |

## 採用条件と再現

- Model：`2toINF/X-VLA-Libero` revision `129e71460678b7236cee6fc9707f09d9fa0c3590`、**追加SFTなし**。batch4学習smokeの重みを評価モデルへ混ぜていない。
- 実行：controller-frame、previous-chunk proprio、30 action/chunk、denoising10、20 Hz、最大300step、frontのみ180度回転。
- 環境：固定LIBERO-plusコード/scene assetsの摂動なし基本8task。開発は初期state index0、確認評価は各task index10〜19。seed・state hash・protocol・source hashを保存。確認評価は調整せず全80件を集計。
- 成功：LIBEROのタスク完了。補助collisionは非操作対象物の安定化後基準から最大L1変位>1mm。公式得点や実機成功率ではない。
- 同数試行なのでmacro成功率と78/80のmicro率は一致。標本数は各task10であり、初期重みの学習データから独立したtask-generalization評価ではない。
- 採用bundle：`models/adopted-base-controller-v1`。37ファイルのhash照合と、ネットワーク無効・export先ソースからの実モデルロード/推論を確認。

根拠：[全体集計](../runs/base-controller-confirmation80/summary.json)、[固定protocol](../runs/base-controller-confirmation80/protocol.json)、[選定動画とhash](../runs/base-controller-confirmation80/success_videos.json)、[評価ソースhash](../runs/base-controller-confirmation80/source_snapshot.json)。各trialのJSON/動画は同じrunフォルダへ保存。

## 実装と検証

- `src/physicalai_vla`：固定資産取得、task照合、HDF5変換/分割、Action Codec、batch4 LoRA SFT、推論、80trial評価/MP4保存/集計、モデルexport、HTTP IPC、期限付きchunk scheduler。
- `ros2_ws/src/vla_msgs`：Observation/ActionChunkメッセージ。colcon build合格、実DDS送受信合格。
- `scripts/ros_nodes.py`：独立ROS PythonからML/sim HTTPサービスを接続。実モデル→HTTP→ROS DDS→simの20秒試験でstatus316件、RUNNING279件。**この試験ではtask成功未確認**。同期80trialの97.5%をROS経由の成功率とは扱わない。
- 学習：batch4でforward/backward・保存、検証8batch、checkpointからstep3への再開まで実行。GPUピーク約10.06 GiB。16episodeの部分データによるsmokeであり、学習性能の判断には使わない。
- W&B：学習entrypointのoffline/disabled実行と、別のonline接続確認を実施。validation loss/LR/epoch相当値/stepを保存。80trial評価はoffline eval run `bvm94edg`へ記録。
- 単体試験15件、依存整合、CLI、Python構文、shell構文を確認。W&Bの非推奨API警告は残るが試験失敗はない。

## 現在の制約

本学習は未実施。state/actionの時間整合・教師再生監査が未完のため、未検証Codecでの長時間SFTをゲートしている。ベースモデルで今回の70%目標を満たしたため、長時間の追加学習は開始していない。

ROS非同期profileはmeasured-proprio・観測age上限1秒・sim時間pauseであり、同期profileと別条件。ROS経由の8task成功率、動画評価、trajectory fusion、実時間保証は未検証。実機制御はユーザー指定により対象外。

評価結果のW&B online送信は、自動承認レビューがローカルprotocol/configを外部へ送る点を理由に拒否した。回避送信せず、ローカルofflineへ保存した。online接続確認runと、この評価payload送信の可否は別である。
