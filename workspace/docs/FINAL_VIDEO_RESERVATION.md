# 最終8タスク連結MP4の実行計画

最終ROS 2評価サービスの終了後、final-ros2-best80から各タスク1episodeずつ収集して自動作成する。

- 出力：`deliverables/final_ros2_eight_tasks_prompted.mp4`（実験job内）。
- 1280×720、H.264 / yuv420p、20fps、CPU encoder 2 threads。
- 左上にモデルへ入力したdataset_instructionの英語文字列をそのまま表示。長い指示は改行だけ行う。
- 各タスクで無衝突成功episodeを確定条件し、その中でepisode番号が最小のものを採用。無衝突成功がない場合に限り、衝突あり成功episodeを採用する。失敗・通信障害・衝突判定不明のepisodeは採用しない。条件に合う成功動画がないタスクはエラーとして記録し、失敗動画で補わない。
- 1episode内を切り貼りせず全frameを保持し、8タスクを指定順に直結する。正方形の元映像は縦横比を維持し、左右を黒で埋める。
- 下部にタスク番号・タスク名・SUCCESS / NO COLLISIONまたはSUCCESS / COLLISION・episode番号。source video/trial/model/protocol hash、prompt、chapter時刻をMP4と同名のJSONへ保存する。
- 全source episodeのframe数=steps+1と、完成MP4のdecode frame数を確認してから完成扱いにする。
- `physicalai-smolvla-final-video.service` が有効化済みで待機中。`video-status.json` と `video-queue.log` に状態を記録。

実装の実動作検証にはbaseline動画を使用し、8本・1,058frames・52.9秒のMP4を生成した。添付画像にはBASELINE PREVIEWと明示している。このpreviewを最終モデルの成果として流用しない。最終動画は学習・ROS 2最終80回評価の完了後に別途生成する。

選択根拠JSONには無衝突成功/衝突あり成功の候補数、選択結果、衝突あり成功へのfallback対象タスクも記録する。無衝突は既存評価の非対象物1mm超変位による判定であり、接触センサーによる実測ではない。
