# 最終ROS 2評価結果

Checkpoint: `${WORKSPACE}/backends/smolvla/runs/burgq-staged-batch32-run/train-002/ckpt-011434`

ROS 2 DDS、2並列、8タスク各10回。初期状態index20〜29。macro成功率 **97.50%**。

インフラエラー: 0件。通信検証を含めた評価有効性: True。

| タスク | 成功 | 成功率 | インフラエラー |
|---|---:|---:|---:|
| milk_basket | 9/10 | 90.0% | 0 |
| tomato_basket | 10/10 | 100.0% | 0 |
| butter_basket | 10/10 | 100.0% | 0 |
| bowl_center_plate | 9/10 | 90.0% | 0 |
| bowl_cookie_plate | 10/10 | 100.0% | 0 |
| bowl_plate | 10/10 | 100.0% | 0 |
| middle_drawer | 10/10 | 100.0% | 0 |
| plate_stove | 10/10 | 100.0% | 0 |

失敗・インフラエラーも分母80に含め、成功するまでの再試行で置き換えていない。実機は未検証であり、この結果は実機でのタスク成功を保証しない。動画とtrial JSONは同じfinal-ros2-best80ディレクトリに保存。
