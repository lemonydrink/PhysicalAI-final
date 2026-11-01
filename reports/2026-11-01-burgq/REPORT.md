# BurgQ SmolVLA × LIBERO-PLUS：8タスク評価・追加学習レポート

記録時刻：2026-11-01T22:31:28.104558+09:00（日本時間）。このレポートは**進行中の実験の中間報告**である。

## 結果

追加学習前の `BurgQ/smolvla-libero-bs32-100k` は、指定8タスク各10回のシミュレーションで **73/80成功、macro成功率91.25%** を記録した。全8タスクで成功動画を保存した。
この値は公開モデルのbaselineであり、今回の追加学習で得た改善としては扱わない。baselineは数値目標70%を上回るが、ユーザー目標の「追加学習したモデルの性能」はまだ確定していない。

SFTはbatch32で進行中。スナップショットの最終記録は **383/11,434 optimizer steps、0.0335 epoch**。最終checkpointの評価値・学習による改善量は未取得である。

## 目的と対象

Franka Pandaのシミュレーションで、牛乳・トマトソース・バターのバスケット収納、ボウル3種の皿への移動、引き出し、皿押しの原案8タスクを実行する。目標は8タスクmacro成功率70%以上と全8タスクの成功動画。実機は使用しない。

作業フォルダは旧X-VLA参照系から発展したためX-VLA/2toINFとROS 2のコードも含む。今回の測定対象はSmolVLAで、ROS 2経由の制御ではなく同期HTTPのシミュレーター接続を使用した。旧モデルの結果は今回のbaselineに混ぜていない。

## モデルとデータ

- 初期モデル：[BurgQ/smolvla-libero-bs32-100k](https://huggingface.co/BurgQ/smolvla-libero-bs32-100k)、revision `238fb0654cd598e287f8b6c2d64bd09af239c76f`。
- 重みSHA256：`88638a6aa1399327d79418265b224b883e7af6b5a7855c3b0aa6b7929ff1f971`。strict読み込みで全500テンソルの値が一致した。
- 公開元の学習データ：[lerobot/libero](https://huggingface.co/datasets/lerobot/libero) @ `a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4`。
- 追加データ：[Sylvest/libero_plus_lerobot](https://huggingface.co/datasets/Sylvest/libero_plus_lerobot) @ `22c57433fef692b5b9ecc0795344daac7fa867a5`。対象8タスク3,071エピソード・404,716フレーム。
- 入力：正面・手首RGB画像、state8D。出力：相対OSC action7D（位置3、回転3、gripper1）。chunk50、10 actionずつ実行。
- normalizerはBurgQ checkpoint付属を維持。学習側の統計で置換しない。

## データ整合監査と修正

全3,071本のstate/actionについて、取得元parquetと学習HDF5の完全一致・finite値・instructionを確認した。3,013本の全操作列が元LIBEROデータの340軌道とbit単位で一致した。残り58本について元軌道との完全一致は主張しない。

各タスクで同一軌道の中間フレームを正面・手首の2カメラ、計16組で比較し、RGB・向きと時刻対応を確認した。画像の照明・背景・視点にはPLUSの拡張差がある。画像監査を全フレームで行ったという意味ではない。
学習画像はidentity、simulatorの生画像は両カメラ180度回転して同じ規約に揃える。state姿勢はrobosuiteのquat2axisangleを用い、座標オフセットを加えない。

元データの時刻メタデータは10Hz、PLUSは20Hzで異なる。ただし全長が等しい同一操作列と、同一フレームで一致するstateを確認したため、保存フレームを間引いたりaction倍率を変えたりしていない。物理的な収録周波数が等しいことまで証明したものではない。

従来のepisode分割では、**検証306本全てに同じ全操作列の軌道が学習側にも存在した**。タスク＋操作列hashによる350グループで分割をやり直した。

| 分割 | エピソード | フレーム | 軌道グループ |
|---|---:|---:|---:|
| 学習 | 2,779 | 365,887 | 316 |
| 検証 | 292 | 38,829 | 34 |
| 合計 | 3,071 | 404,716 | 350 |

完全一致グループの分割間重複は0。整列prefixのMAE<0.005または完全一致フレーム割合>50%という近似重複検査でも分割間0組だった。全ての時間ずれ・意味的な類似軌道の不存在を保証する検査ではない。
BurgQは元LIBEROを事前学習済みなので、この検証splitを「事前学習でも未使用」とは扱わない。
詳細：[監査記録](audit/dataset_alignment.json)、[実装側の監査説明](../../workspace/backends/smolvla/BURGQ_DATA_AUDIT.md)。

## 学習設定

| 項目 | 設定 |
|---|---|
| 学習対象 | Action Expert＋state/action projections、99,880,992 trainable parameters |
| VLM・vision | 凍結 |
| 数値精度 | trainable重みFP32、BF16 autocast |
| batch / accumulation | 32 / 1、effective batch32 |
| optimizer | AdamW、betas(0.9,0.95)、eps1e-8 |
| LR | peak3e-5、warmup572更新、cosine decay、最終3e-6 |
| weight decay / clipping | 1e-4 / gradient norm1.0 |
| 学習量 | 学習splitを1巡、365,887フレーム、11,434更新 |
| loss | 有効な7 action次元と有効な時刻のみ、末尾paddingは除外 |
| checkpoint / rollout | 2,000更新ごと＋最後、8タスク×10回 |
| validation | 中間は各タスク8サンプル、最後は検証split全フレーム |
| 選択 | rollout総成功数、同点時は最低タスク率・成功したタスク数・早いstepの順 |
| tracking | W&B offline、設定・train loss・validation・LR・epoch/step・rolloutを記録 |

監査済みconfigとmanifestのSHA256を学習エントリポイントが照合する。checkpointの継続ではデータ順・optimizer等を保持するが、今回のBurgQ開始時は旧実験の重みやoptimizerを継承せず新規開始した。

## baseline評価プロトコル

摂動なしの基本8タスク、各10試行、最大300step。初期状態index10〜19、seedは42＋dataset task index×100000＋試行番号。action_dt0.05秒、相対OSC、モデルのnormalizerで逆変換して[-1,1]にclipする。同じ80試行で学習checkpointを比較する。
LIBERO-PLUSの全摂動に対する網羅的評価ではなく、公開モデルの論文・model cardのスコアとの直接比較も意図しない。

| タスク | 成功 | 成功率 | Wilson 95%区間 |
|---|---:|---:|---:|
| milk_basket | 8/10 | 80% | 49.0%–94.3% |
| tomato_basket | 9/10 | 90% | 59.6%–98.2% |
| butter_basket | 10/10 | 100% | 72.2%–100.0% |
| bowl_center_plate | 9/10 | 90% | 59.6%–98.2% |
| bowl_cookie_plate | 8/10 | 80% | 49.0%–94.3% |
| bowl_plate | 10/10 | 100% | 72.2%–100.0% |
| middle_drawer | 9/10 | 90% | 59.6%–98.2% |
| plate_stove | 10/10 | 100% | 72.2%–100.0% |
| **全体** | **73/80** | **91.25%** | — |

各タスク10試行のため区間は広い。全試行が揃い、infra errorは0、全タスクに成功動画がある。
補助的なcollision-free成功は60/80（75%）で、plate_stoveは0/10だった。タスク完了成功率と衝突指標を混同しない。実機の安全性を測定した結果ではない。

## 成功動画

baseline成功動画を各タスク1本ずつ [videos/](videos/) に収録し、[videos.json](videos.json) に元trialとhashを記載した。連続rolloutであり、成功動作を複数試行から合成していない。動画は追加学習前の公開モデルのもの。

## 制約・未完了事項

1. SFT完了後の性能とbaseline差は未確定。学習lossだけで性能改善を主張しない。
2. 同じ80回でcheckpointを選択するため、選択後の独立な最終確認評価ではない。
3. 事前学習データとの軌道重複がある。新規タスクへの汎化を検証していない。
4. 公開フォルダには重み・学習データ・シーンassets・venv・W&Bバイナリを含めない。実行には別途取得が必要。
5. 実験plan・manifestには実施環境の絶対パスが証跡として残る。別PCでは再配置・再監査が必要で、そのまま一発実行できる完成済みパッケージとはしていない。

## 検証と追跡

既存backendテスト12件、依存関係チェック、lock整合、構文、学習CLIを確認済み。公開コピーでもcoreテスト15件、資産不要のbackendテスト8件が成功。重み・rawデータを必要とする4件は公開コピーでの対象外とした。[公開検証記録](PUBLICATION_VALIDATION.json)を参照。
設定・生trial JSON・監査・学習途中のmetrics・各ファイルhashを同梱した。[再現手順と必要資産](../../REPRODUCIBILITY.md) を参照。

この公開版は記録時点のスナップショットであり、学習プロセスや完了通知機能をGitHubへ移したものではない。
