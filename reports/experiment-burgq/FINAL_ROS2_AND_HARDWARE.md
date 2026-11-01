# 最終80回評価と実機移行の実装範囲

現在の実験はシミュレーション専用。実機での成功を確認したという意味ではない。

## 実行計画した最終評価

1. 学習・途中checkpoint評価を最後まで完了する。途中のHTTP評価は4並列。
2. rolloutで選んだbest checkpointを固定し、学習サービスがinactiveになるまで待つ。GPU上の学習プロセスやoptimizerを保持したまま最終評価を重ねない。
3. 別プロセスでシミュレーター2台、ROS 2 gateway/robot node各2個、共有SmolVLAモデル1個を起動する。
4. 8タスク各10回、計80回をROS 2 DDS経由で実行。選択用index10〜19と別のindex20〜29を使う。全タスク50初期状態ありを確認済み。
5. 成功率、非対象物変位によるcollision-free成功率、通信エラー、動画、protocol、モデルhash、DDS action数を保存しW&Bへ記録する。

最終80回の出力は `backends/smolvla/runs/burgq-staged-batch32-run/final-ros2-best80/`。
`RESULT.md` と `final_ros2_result.json` は完了時に生成する。現時点で最終性能は未確定。
`physicalai-smolvla-final-ros2.service` は有効化済み。学習・モデル選択の正常終了とプロセス退出を待機する。終了済みの学習や評価を推測で完了扱いにはしない。

最終評価の初期状態はselection用とは分けるが、元LIBEROが事前学習に含まれる点は変わらない。HTTP baseline91.25%との単純差は通信方式と初期状態の差も含む。

## 実際の通信経路

```text
SmolVLA evaluator (ML Python)
  → local HTTP interpreter adapter
  → ROS 2 gateway node
  → DDS ActionChunk (7D relative OSC, horizon1)
  → ROS 2 robot node
  → simulator/robot-driver adapter
  → DDS Observation (front/wrist RGB, measured state, timestamps)
  → gateway → evaluator
```

HTTPはPython環境間のローカルIPCとdriver境界に残る。観測とactionが実際にDDSを通る。ROS 2 nodeを起動するだけの見せかけの経路ではない。
旧 `scripts/ros_nodes.py` はX-VLAのabsolute10D参照系。SmolVLAは新しい `scripts/smolvla_ros_transport.py` と `smolvla-relative-osc7-v1` codecを使用し、混用しない。

各trial JSONにROS namespaceとDDS受信action数を記録する。正常trialはDDS action数とシミュレーションstep数の一致を検査する。通信障害はinfra_errorとして失敗に数え、分母80から除かず、同じtrialを成功するまで再試行しない。最終評価の受理にはinfra errorなしも要求する。

## 入力・停止・復旧

- episode、指令sequence、checkpoint SHA256、codec、7次元shape、finite値、[-1,1]、action_dt0.05秒を確認。
- actionが参照する観測timestampと現在の観測が一致し、同一ホストmonotonic clockで2秒以内であることを検査。
- 期限切れ、重複、モデル不一致、形式不正ではfaultをラッチし、driver `/stop` を呼ぶ。明示的なresetまで実行を再開しない。
- シミュレーターは停止後の `/step` を拒否し、状態を進めない。シミュレーション時間停止を実機の非常停止と同一視しない。
- reset/action/observationの各worker topicをnamespaceで分離。最終評価はROS_DOMAIN_ID63、worker_0/worker_1。
- 同じホストの時計を前提にする。複数PCへ移す場合は時刻同期とtimestamp変換の再検証が必要。

## 実動作の確認範囲

2台のMuJoCo環境で、各3 actionがDDSを通ること、期限切れaction拒否、停止ack、reset後の復旧を検証した。
さらに実際のBurgQ由来SmolVLA checkpointで2並列・各5step（計10 action）のROS 2 rolloutを確認した。これは配線・制御経路のsmokeであり、5stepでタスク成功したという評価ではない。smokeはindex0/1を使い、最終80回のpolicy評価には流用しない。
coreテスト32件、backend14件、依存関係・lock・構文・CLIを確認。

## 実機に向けて追加したもの

`src/physicalai_vla/hardware_adapter.py` のMeasuredCartesianAdapterは、校正済みdriverの測定位置・姿勢を基準に相対OSCをCartesian setpointへ変換する。

- policy座標系からrobot baseへの回転、実機で確定した位置/回転gainを使う。
- 測定feedbackの鮮度・姿勢・collisionを確認。古い推定poseでopen-loop実行しない。
- 作業範囲と並進/角速度上限、gripper開口幅・速度上限を適用する。
- command sequenceとcontroller acknowledgementを確認。異常時はcontroller.stopを呼んでfaultをラッチする。
- 加速度/jerk制限・IK/関節制限・独立watchdog・非常停止は実機の下位controllerが担う必須条件。PythonのROS nodeをハードリアルタイムservoに見立てない。

模擬controllerで速度/開口幅制限、古い測定・collision・作業範囲外での停止を単体検証した。実機で検証したという意味ではない。

## 実機でまだ必要なもの

Frankaの型式・firmware・driver/control interfaceが未指定で実機未準備のため、vendor固有のdriver bindingと実機commissioningは完了していない。
実機接続には、実測の2カメラ・robot base/TCP外部校正、カメラ向き、8D stateとaxis-angleの規約、controller-frame/単位/gain、gripper符号・幅、作業範囲、速度/加速度、collision/非常停止、20Hz上位処理と下位servoの期限・停止挙動を確認する必要がある。
シミュレーターの位置gainを無条件に実機へコピーしない。制限で動作が変わればpolicyの閉ループ性能も実機で再評価する。

実機adapterの `/health` はhardware modeとcommissioning済みcapabilitiesを返し、実測観測を `/reset`・`/step` で返し、`/stop` を確実に実装する必要がある。
ROS robot nodeは `--allow-commissioned-hardware` が指定され、必要capabilityが全て確認された場合だけhardware adapterを受け付ける。今回の実行計画はこのフラグを付けずsimulationだけを起動する。

したがって現時点の成果は「実機移行を想定した制御境界・ガードと、ROS 2によるシミュレーション制御」である。「このまま実機に接続すれば必ず8タスクを遂行する」という主張はしない。
