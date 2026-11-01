# PhysicalAI-final

[最終レポートPDF（3ページ）](reports/experiment-burgq/final-report/REPORT_3P.pdf)
BurgQ SmolVLAをLIBERO-PLUSの指定8タスクへ追加学習するシミュレーション実験。

**学習・最終評価・動画生成は完了しました。**

- モデル比較（同じ初期状態・HTTP）：73/80（91.25%）→78/80（97.5%）、+6.25ポイント。checkpoint選択用セットでの観測値。
- 最終ROS 2評価（別初期状態・2並列）：78/80（97.5%）、無衝突成功68/80（85.0%）、インフラエラー0。通信方式による改善量とは扱わない。
- 8タスクすべて無衝突成功episodeを収録した、プロンプト付き48.65秒のMP4が完成。実機は未検証。

[最終レポート：モデル性能とROS 2実行を分けて評価](reports/experiment-burgq/final-report/REPORT.md)

[完成MP4](reports/experiment-burgq/final-report/assets/final_ros2_eight_tasks_prompted.mp4)

- [4台並列評価への切替記録](reports/experiment-burgq/PARALLEL_EVALUATION.md)
- [最終ROS 2評価の実行計画と実機移行範囲](reports/experiment-burgq/FINAL_ROS2_AND_HARDWARE.md)
- [プロンプト付き8タスク連結MP4の実行計画](reports/experiment-burgq/FINAL_VIDEO_RESERVATION.md)
- [旧中間レポート（経緯を保存）](reports/experiment-burgq/REPORT.md)
- [8タスクの成功動画](reports/experiment-burgq/videos/)
- [設定・監査・80試行の結果](reports/experiment-burgq/)
- [作業フォルダ](workspace/)
- [再現手順・公開範囲](REPRODUCIBILITY.md)

公開時点のコード・依存lock・ROS 2メッセージ定義・過去の設計記録を保存しています。重み、データセット、シーンassets、仮想環境、秘密情報は含みません。`workspace/` 内の旧ドキュメントには過去の設定や結果が残っています。現行実験の説明は上記レポートを基準にしてください。

動画採用8タスクの[LeRobot形式データと軌道可視化](reports/experiment-burgq/final-report/trajectories/README.md)を公開しています。
