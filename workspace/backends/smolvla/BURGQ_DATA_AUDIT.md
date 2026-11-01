# BurgQ × LIBERO-PLUS データ整合監査（2026-11-01）

判定：追加学習に使用できる。検証分割の重複を修正し、監査済みmanifest/configのSHA256が一致する場合だけ学習を許可する。

## 対象と根拠

BurgQ/smolvla-libero-bs32-100k @ 238fb0654cd598e287f8b6c2d64bd09af239c76f。
公開元が示す学習データ lerobot/libero @ a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4 と、追加学習データ Sylvest/libero_plus_lerobot @ 22c57433fef692b5b9ecc0795344daac7fa867a5 を比較した。
元データ全377 parquet（273,465フレーム）と比較用16動画を取得。取得したLFSファイル395件のSHA256を照合。

## 確認結果

| 項目 | 結果 |
|---|---|
| 操作の意味・順序 | 相対OSC、位置3＋回転3＋グリッパー1。3,071本中3,013本で元データの全操作列とbit単位で一致（元軌道340本） |
| state | 位置3＋axis-angle3＋指関節2。タスクごとの高さ基準も元データと一致。座標オフセットは加えない |
| 数値の変換 | 全3,071本・404,716フレームのstate/actionが取得元parquetとHDF5で完全一致、NaN/Infなし |
| タスク条件 | 全HDF5のinstructionが指定8タスクの文字列と一致。データセット間でtask_indexは異なるため文字列で対応 |
| 画像 | 各タスクの同一軌道・中間フレームで2カメラ計16組を比較。RGB・256×256・向きは同じ。元画像に回転や反転を追加しない |
| 時間対応 | 8タスクともstateは同一フレーム対応が最良。±1フレームずらすと誤差増大 |
| 時刻メタデータ | 元は10Hz、PLUSは20Hz。しかし同一長の操作列・stateが同じ順序で対応。フレーム間引き・複製・action倍率変更はしない |
| 正規化 | BurgQ付属の8D state / 7D action mean/stdを維持。PLUS統計で上書きしない。往復・7D loss・padding勾配ゼロをGPU preflightで確認 |
| シミュレーター入力 | 学習データはidentity、simulator生画像は両カメラ180度回転して同じ向き。姿勢はrobosuite quat2axisangle |

画像の照明、背景、正面カメラの視点はPLUSの拡張で変わる。画像が画素一致するという判定ではない。
画像比較の対象は16組であり、全動画の全フレームと元データの画像を比較したわけではない。
元データと操作列の完全一致を確認できなかった58本も、取得元との一致・数値仕様・instructionは全件検査した。

## 発見した問題と修正

従来のepisode単位分割では、検証306本の全てについて、同じ全操作列の軌道が学習側にも存在した。見た目の違う拡張エピソードが分割をまたいでいたため、検証lossに情報漏洩があった。

タスク＋全操作列SHA256を元に350軌道グループを作り、seed42でタスクごとに約10%のグループを検証側へ割り当てた。同一操作列グループの分割間重複は0。整列した先頭部分の近似重複（MAE <0.005 または完全一致フレーム割合 >50%）も分割間0組。

- 学習：2,779エピソード、365,887フレーム、316グループ。
- 検証：292エピソード、38,829フレーム、34グループ。
- 合計：3,071エピソード、404,716フレームを保持。
- 新manifest：data/grouped-v2/manifest.json。既存HDF5を参照し、画像や操作は変更しない。
- batch32・勾配蓄積1で1epoch = 11,434 optimizer steps、warmup572。末尾の実データ31件はpaddingをlossから除外。
- 訓練統計も新splitから再集計したが、実際のnormalizerはBurgQ付属を使う。

BurgQの事前学習自体に元LIBERO軌道が含まれるため、この検証を事前学習でも未使用のデータとは扱わない。checkpointの選択には8タスク各10回のrollout成功率を用いる。同じ80回での選択は独立した最終評価ではない。

## 実行への接続

plan.jsonのtraining_manifest_pathで新splitを使用。evaluation_manifest_pathは既存baselineのprotocolを維持するため旧manifestを固定する。rolloutではHDF5学習窓を使わずBurgQ自身のnormalizerを使うので、評価入力は変わらない。

staged_training.pyはaudit/dataset_alignment.jsonのpassedと、実際のconfig・学習manifestのSHA256を照合してから実行する。baseline80を完了してから新規optimizerでSFTし、2,000更新ごとと最後に80回rolloutする。W&B offline記録を維持。

## 検証と証跡

backend既存テスト12件成功。uv pip check、uv lock --check、構文、学習CLIを確認。
詳細は runs/burgq-staged-batch32-20261101/audit/ 以下：

- dataset_alignment.json：最終判定と固定hash、制約。
- reference_numeric_comparison.json：全操作列の照合とタスク別分布。
- reference_images_temporal.json / reference_image_pairs.jpg：画像16組と±1フレーム比較。
- raw_conversion_verification.json：全件の取得元との数値一致。
- group_split.json / near_duplicates.json：分割監査。
- contract.json：初期重み全500テンソル一致、normalizer往復、7D lossのGPU確認。
- source/：監査スクリプト。
