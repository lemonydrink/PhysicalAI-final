# 未確定事項・意思決定台帳

2026-10-30／v0.4／D-02は実装系統を確定、資産復元等は未完。推奨案は確定・承認・検証済みを意味しない。

調査で確定できる項目は実装担当が証拠を採取する。機材・目的・実機運用条件は所有者と確認する。担当者名・期限は計画時に割り当てる。各項目の閉鎖時に決定日、採用案、棄却理由、証拠パス、影響要件・試験を追記する。

| ID | 未確定事項 | 推奨案・選択肢 | 確定のための証拠 | 閉鎖期限 / 判断者 |
|---|---|---|---|---|
| D-01 | dataset実在ID、revision、8task対応 | 原案repoを調査しtask manifest固定 | metadata実体、指示照合、episode数、hash、suite対応 | P1前 / 実装担当 |
| D-02 | 初期checkpointとaction方式 | LIBERO＋絶対targetを先行調査。変換不成立ならbase＋相対action | config/processorとloader、教師target再生、比較baseline定義 | 学習開始前 / 研究担当 |
| D-03 | Codec詳細と許容誤差 | frame、scale、回転、gripper、domain IDを実装から採取 | controller目標生成式、fixture、残差分布と数値閾値 | P1完了前 / 実装担当 |
| D-04 | GPUと学習/推論構成 | 16 GBは原案想定。実測後にbatch/chunk等を固定 | GPU driver/CUDA、VRAM/RAM、1 batch・推論p95/p99、保存復元 | P0〜P2前 / 実装担当 |
| D-05 | LeRobot/LIBERO/ROS環境境界 | 別環境＋IPC adapterを提案 | lock、ABI、画像転送経路、container/IPC smoke | P3前 / 実装担当 |
| D-06 | Panda機材互換 | robot型式・serial・FW・FCI・libfranka・franka_ros2・ROS/OS/RT kernelを一組で固定 | 機材台帳、メーカー互換情報、fake/接続試験 | P5前 / 機材所有者＋実装担当 |
| D-07 | Controller・IK | 対応する既存controller優先。独自プラグインは必要性確認後 | command/state interface、更新周期、IK/特異点/衝突検証、停止挙動 | P5前 / 制御担当 |
| D-08 | 安全limitsと停止profile | 原案値は試験候補。加速度・jerk・外力・workspace等を追加 | 数値limits、減速開始/完了期限、最大停止距離、故障試験結果 | real起動前 / 制御担当＋運用責任者 |
| D-09 | 同期と遅延予算 | 20 Hz行動、inflight=1、最新待機1。5 Hzは実測で判断 | clock同期誤差、sensor skew、stage latency、timeout余裕 | P4完了前 / 実装担当 |
| D-10 | 評価protocolと成功定義 | 8task macro、paired比較、接触と衝突を区別 | task/variant/初期状態一覧、成功判定、CI計算法、max_steps、採否規則 | P2本評価前 / 研究担当 |
| D-11 | カメラ・治具・記録運用 | front/wrist＋校正済みTCP、把持・配置から開始 | 機材型番、intrinsics/hand-eye、治具図、保存容量/期間、gripper停止方針 | P5〜P6前 / 機材所有者＋運用責任者 |
| D-12 | 実機追加SFT | 必要時に各task 10〜30軌道を収集開始案 | 実機baseline失敗分析、収集条件、混合比、分割、改善比較 | 追加学習前 / 研究担当＋所有者 |

## PARC照合による追加項目

| ID | 未確定事項 | 推奨案・選択肢 | 確定のための証拠 | 閉鎖期限 / 判断者 |
|---|---|---|---|---|---|
| D-13 | 8タスク選定の意図 | 原案の新集合を候補保持。PARC T2継続なら過去集合へ変更 | 原案と過去mapは共通indexなし。対象集合の判断とrevision付きmanifest | データ抽出前 / 研究担当＋所有者 |
| D-14 | 過去モデルの時間軸 | raw 20 Hzとhandler 30 Hzを比較、今回の実行dtを再定義 | raw stamp→教師窓→モデルtarget→controller実行の追跡と再生試験 | Codec確定前 / 実装担当 |

D-01：過去dataset revisionが判明。新8taskの実体検証は未完。D-02：過去はupstream版で新原案はLeRobot版。過去base保存先は現在欠落。D-03：絶対target変換実装と限定fixtureあり。rot6D helperの異なるレイアウトを監査する。D-04：過去batch4の学習実行記録あり、現在のGPU/LeRobot性能は未確定。D-10：過去holdoutなし、best_eval未選択。詳細は[照合書](PARC_CROSSCHECK.md)。これらの証拠追加だけでは各判断をDECIDEDにしない。

## 初回レビューで確認する事項

1. 当面の到達点をシミュレーション8タスクまでとするか、実機までを同一計画で進めるか。
2. Panda、FCI、カメラ、治具の利用可否と型式。未所有ならP5以降を将来工程とする。
3. 通常環境80%・ROS劣化5 points等の目標を採用するか、予備実験後に調整するか。

現時点では回答待ちで文書作成を止める必要はない。これらは該当工程に入るまでに閉鎖する。

## 決定記録テンプレート

```text
ID:
状態: OPEN / INVESTIGATING / DECIDED / DEFERRED
決定日:
判断者:
採用案:
代替案と棄却理由:
根拠のrevision・hash・証拠パス:
影響する要件ID・受け入れ試験ID:
再検討条件:
```

## 2026-10-30 ユーザー指定による更新

- D-02：**採用系統はDECIDED：2toINF upstream X-VLA**。LeRobot版比較は必須から除外。過去base重みの復元、hash、既存adapterの再ロードは引き続きOPEN。過去revisionを復元の第一候補とする。
- D-10：**成果の種類はDECIDED：実機のタスク成功動画**。元の成功率閾値・30試行評価は任意。動画の合格条件はREQUIREMENTS第0節DEMO-01〜03/AT-DEMOに従う。
- D-13：撮影対象taskと件数はOPEN。まず1つの把持・配置taskで完了動画を得る経路を提案し、8task全部を必須と解釈しない。
- D-04：新規SFTが必要と判断された場合だけ学習資源を調査する。推論GPUとモデルロードの確認は必要。

上記は旧表の推奨案より優先する。安全な実機接続に必要なD-03、D-06〜09、D-11、D-14は維持する。

## 最新更新：成功率70%・全8task撮影

この更新は上記の動画のみを成果とする記述に優先する。

- D-10：**成功率70%というモデル目標と8taskすべての撮影はDECIDED（ユーザー指定）**。測定は実機macro平均を提案。各task最低70%とするかは未確定。試行数は各10回/計80回を提案し、評価前に固定する。真の成功率の信頼下限70%という要求は受けていない。
- D-13：**撮影task数=8はDECIDED**。原案8taskと過去T2のどちらを対象にするかはOPEN。まず1taskで経路検証するが、それを納品範囲とはしない。
- SFTと追加データ収集は70%へ改善する手段として必要に応じて実施する。単一凍結bundleの最終評価と各task成功動画の両方を必須にする。
- 合格判定と納品内容はREQUIREMENTS v0.4第0節GOAL-01〜04/DEMO-01〜03/AT-GOAL/AT-DEMOを参照する。

## 予選追加照合による更新

予選報告とbody-frame版ローカル評価に100k LoRAの20/20成功記録が見つかった。初期学習baselineは予選LoRA＋Soft Prompt＋Action層、batch2へ変更する。D-03/D-14にはbody/controller frameとaction時刻の食い違い検証を追加する。別経路の35%診断を含め、成功した推論境界と学習条件を分離せず復元する。予選成功を実機8task70%の保証とは扱わない。詳細はPARC_CROSSCHECK F-14〜17。

## 学習batch確定（2026-10-30）

ユーザー指定により今回の学習batch size=4をDECIDEDとする。過去予選batch2は履歴として維持する。単一GPU・gradient accumulation=1（実効batch4）は初期構成案。D-04ではbatch4の資源使用と保存復元を検証し、OOM時も無断でbatchを変更しない。この指定は過去のbatch2を採用する推奨より優先する。

## 実装開始時のユーザー回答（最優先）

- D-13：DECIDED。原案の8タスクを採用し、PARC本選T2集合は使わない。
- 実機は使用しない。目標成功率70%と全8task動画をシミュレーションで取得する。D-06〜08/D-11の実機固有部分は今回対象外。
- D-01：固定revisionのmetadata取得済み。原案8task index/指示は全件一致。元の集計も14,347episode、2,238,036frame、20Hz、40task、v2.1と一致。data/tasks.resolved.jsonにsuite・BDDL・init state hashを保存。
- D-05：学習/推論/sim専用uv環境とROS専用Pixi環境をHTTP IPCで分離する。

## SmolVLAへの主系統変更（2026-10-30、最優先）

SmolVLAへ切り替える。batch4、原案8task、sim完結、追加学習成果と未調整比較を維持する。汎用smolvla_baseを初期点とする案を採用して実装。旧2toINF採用の判断は履歴へ変更し、X-VLAの78/80を今回の学習成果に数えない。PARCのSmolVLA記録はproxy loss探索であり、70%性能を実証したものではない。公式モデルカードを初期点の参照とする：https://huggingface.co/lerobot/smolvla_base
