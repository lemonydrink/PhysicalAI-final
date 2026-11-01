# 完了結果・編集済み動画（2026-11-02）

学習は365,887フレームを1巡し、11,434更新で完了。best checkpointは11,434更新。
最終評価はROS 2 DDS・2並列・初期状態index20〜29で実施し、78/80成功（97.5%）、既存の非対象物変位指標による無衝突成功68/80（85%）、通信エラー0件。実機は未検証。

| タスク | 成功 | 無衝突成功 |
|---|---:|---:|
| milk_basket | 9/10 | 9/10 |
| tomato_basket | 10/10 | 10/10 |
| butter_basket | 10/10 | 10/10 |
| bowl_center_plate | 9/10 | 9/10 |
| bowl_cookie_plate | 10/10 | 5/10 |
| bowl_plate | 10/10 | 10/10 |
| middle_drawer | 10/10 | 10/10 |
| plate_stove | 10/10 | 5/10 |

[編集済みMP4](final-deliverables/final_ros2_eight_tasks_plain_intro5_pause1.mp4)

8本とも無衝突成功のepisodeを採用。左上に実際のpromptを表示する。動作部分は20fps・1倍速。冒頭に実験名と8タスク一覧を5秒、タスク間7か所に完了frameの1秒holdを追加した。元の48.65秒から60.65秒（1,213 frames）になった。episode内の間引き・速度変更・途中の切り貼りはない。

同名JSONにsource hash・trial・prompt・chapter開始/終了・hold frame数を保存した。

baselineの73/80は別の初期状態index10〜19・HTTP経路による評価。最終値との差だけで学習の因果効果を断定しない。元の中間レポートは当時のsnapshotとして保存する。

冒頭画面は白背景・黒文字・serif書体の簡素な研究発表風の実験名と8タスク一覧に更新。冒頭5秒、タスク間7×1秒、全長60.65秒は維持。
