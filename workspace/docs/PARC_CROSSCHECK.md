# PARC既存事実との照合 v0.1

照合日：2026-10-30／対象snapshot：PARC2026_final HEAD `eeddacf7b79bb2f9089b6a013410f0a0913f7ea2`（照合開始時git statusはclean）。既存PARCファイルを変更せず参照した。

## 判定ルール

- **実行記録**：ログ・state・成果物が残る。現在の再現成功や方策性能まで意味しない。
- **コード/設定**：実装に存在する。実データ・実機での正しさは別途検証する。
- **運用契約**：PARCで定めた比較・環境ルール。物理法則や実験結果ではない。
- **未確認**：今回確認した資料から結論できない。

`experiment.json`のplanned/runningやREADMEだけで履歴を判定せず、runとcheckpointの実体を併読した。現在のX-VLA/ROS 2計画への適用可否を以下に示す。元原案は変更しない。

## 1. タスク集合は一致していない（F-01）

**コード/設定**：PARC T2の公開dataset indexは`[22, 1, 12, 35, 34, 29, 23, 32]`。新原案の`[4, 6, 7, 8, 9, 16, 20, 36]`とは共通要素がない。原案の「前回選定した8タスクを維持」はPARC T2との関係では成立しない。

根拠：[公開task対応map](../../PARC2026_final/experiments/T2-Xvla-libero/code/dataset/prepare_t2_source.py)、[T2 task CSV](../../PARC2026_final/experiments/T2-Xvla-libero/config/T2_TASKS.csv)。CSVの非公開環境詳細を本書に複製しない。

**反映**：新原案8タスクを別の候補セットとして保持する。PARC T2継続を意図する場合は選定を変更する必要がある。D-13として意図確認を追加し、学習データ抽出前に決める。新8タスクのindex/指示が正しいことは過去T2 mapでは検証できない。

## 2. モデル系統・revisionが異なる（F-02）

**コード/設定＋実行記録**：過去T2の初期重みは`2toINF/X-VLA-Libero`、revision `129e71460678b7236cee6fc9707f09d9fa0c3590`。新原案は`lerobot/xvla-libero`である。同じモデル名でも実装・processor・checkpoint互換を仮定しない。

根拠：[model config](../../PARC2026_final/experiments/T2-Xvla-libero/config/model.json)、[実行manifest](../../PARC2026_final/runs/T2-Xvla-libero/20260903T152306Z/run.json)。

**反映**：D-02に「過去upstream版再現」と「LeRobot版移行」の比較を追加する。過去LoRAをLeRobot版へそのままロードする経路は承認済み互換経路ではない。

## 3. データ源と分割（F-03）

**コード/設定**：過去も`Sylvest/libero_plus_lerobot`、固定revision `22c57433fef692b5b9ecc0795344daac7fa867a5`、source fps=20。過去選択は`holdout=false`で、今回の90/10 validation分割とは異なる。過去manifest hashは`13d771ec56c6ccfbbc063e724b9c52d1b06defa59d0ca91a8bf2541c9e5838b5`とrun manifestに記録されている。

根拠：[data config](../../PARC2026_final/experiments/T2-Xvla-libero/config/data.json)、[run manifest](../../PARC2026_final/runs/T2-Xvla-libero/20260903T152306Z/run.json)。

**反映**：dataset IDは未知から「過去固定revisionあり」に更新。ただしraw実体・全件統計・新8タスクの内容は未再検証。過去全件学習済みモデルを用いる場合、新設validationが初期重みの学習から独立とは保証できない。SFTのtrain/val非重複とpretraining汚染を分けて報告する。

## 4. 絶対target変換は既存実装がある（F-04）

**コード/今回単体試験**：`_reconstruct_absolute_actions`は`state[j-1]`と`action[j]`（j≥1）を対応させ、最初のactionを落とす。位置scale=0.05 m、回転scale=0.5 rad、回転は`R_delta @ R_previous`。最初の6D actionを[-1,1]へclipし、出力はxyz3＋rot6D6＋raw grip1の10D。

根拠：[変換器](../../PARC2026_final/experiments/T2-Xvla-libero/code/dataset/prepare_t2_source.py)、[fixture](../../PARC2026_final/experiments/T2-Xvla-libero/tests/test_t2_xvla_layout.py)。

**反映**：「変換方法が全く未解決」ではなく「候補実装とfixtureあり、controller/教師再生照合が必要」にする。単体試験はゼロ回転等の限定例であり、非可換回転・全8task教師再生の証拠ではない。raw stateの後ろ2成分や実機stateの意味まで本関数から確定しない。

## 5. モデル内20Dと有効片腕10D（F-05）

**コード/設定**：過去`ee6d` action spaceは20D、gripper index 9/19。LIBERO handlerは左10D、右zeroを構成し、domain IDは3。過去configの`model_output=absolute_ee6d_10`は有効片腕部分の契約であり、内部tensorが常に10Dという意味ではない。

根拠：[action hub](../../PARC2026_final/experiments/T2-Xvla-libero/code/training/models/action_hub.py)、[LIBERO handler](../../PARC2026_final/experiments/T2-Xvla-libero/code/training/datasets/domain_handler/simulations.py)、[domain map](../../PARC2026_final/experiments/T2-Xvla-libero/code/training/datasets/domain_config.py)。

**反映**：D-03の初期証拠に追加。LeRobot版でdomain/packing/maskが同一かは別途照合する。

## 6. 20 Hz sourceと30 Hz handlerの差（F-06）

**コード/設定**：source fps=20に対し、`LiberoHandler.build_left_right`は`freq=30.0,qdur=1.0`。共通handlerは時刻をindex/freqから作り、未来1秒を`num_actions+1`点へ補間する。過去model configのnum_actions=30と組み合わせると、sourceの実時間と同じ20 Hz系列として解釈できない。

根拠：[source fps](../../PARC2026_final/experiments/T2-Xvla-libero/config/data.json)、[LiberoHandler](../../PARC2026_final/experiments/T2-Xvla-libero/code/training/datasets/domain_handler/simulations.py)、[共通時間処理](../../PARC2026_final/experiments/T2-Xvla-libero/code/training/datasets/domain_handler/base.py)。

**反映**：D-14で教師窓・補間時刻・モデル出力dt・実行dtを照合する。30点だから1.5秒、32点だから1.6秒とcheckpointの意味を確認せず決めない。これは差の発見であり、過去学習失敗の原因と断定したものではない。

## 7. 色・画像先頭・カメラ姿勢（F-07）

**コード**：変換器はRGB配列をそのままPNG encodeし、既存decoderがBGR→RGB変換をしない挙動と数値対応させている。LIBERO handlerは画像先頭を1枚落とす。推論adapterはagentviewだけ180度回転、wristは回転しない。controller姿勢がない場合にbody→controller固定回転を用いる。

根拠：[PNG処理とalignment](../../PARC2026_final/experiments/T2-Xvla-libero/code/dataset/prepare_t2_source.py)、[decoder](../../PARC2026_final/experiments/T2-Xvla-libero/code/training/datasets/utils.py)、[推論adapter](../../PARC2026_final/experiments/T2EXP-005-xvla-libero-improved/code/inference/policy_impl.py)。

**反映**：カメラ名を置換するだけで移植しない。2カメラ色fixture・時刻alignment・各view回転・TCP/controller frameをgolden入力で比較する。実機カメラへこの180度回転や固定回転を無条件適用しない。

## 8. rot6Dレイアウトに監査対象がある（F-08）

**コード**：今回確認した変換器とaction_adapterは`[col0の3成分, col1の3成分]`。一方trainingの`datasets/utils.py`には`matrix[..., :, :2].reshape(...)`のrow-major flattenと`0:5:2`/`1:6:2`で復元する関数がある。helperの存在だけで本LIBERO経路に混在しているとは断定できないが、同じrot6Dという名前で相互置換できない。

根拠：[変換器](../../PARC2026_final/experiments/T2-Xvla-libero/code/dataset/prepare_t2_source.py)、[action adapter](../../PARC2026_final/experiments/T2EXP-005-xvla-libero-improved/code/common/action_adapter.py)、[utils](../../PARC2026_final/experiments/T2-Xvla-libero/code/training/datasets/utils.py)。

**反映**：D-03で呼出経路を特定し、非自明な回転の往復fixtureで検証する。関数名一致やidentityだけで互換判定しない。

## 9. 学習実行はある。Soft Prompt単独の実績とは異なる（F-09）

**実行記録**：T2-Xvlaのbatch4 preflight（step1）、本学習checkpoint stateのglobal_step=82468、終盤train logが残る。improvedのcheckpoint_indexには82468/93033があり、train logに93033/93033と保存完了が残る。

improvedのparameter_summaryは総879,482,456、更新11,868,760（約1.35%）。内訳はVLM LoRA 4,849,664、policy LoRA 3,178,496、Soft Prompt 983,040、action 2,857,560。Soft Promptのみの学習ではない。

根拠：[step1記録](../../PARC2026_final/runs/T2-Xvla-libero/preflight-batch4/ckpt-1/state.json)、[step82468記録](../../PARC2026_final/runs/T2-Xvla-libero/20260903T152306Z/ckpt-82468/state.json)、[checkpoint index](../../PARC2026_final/runs/T2EXP-005-xvla-libero-improved/production-two-phase-20260904-r2/checkpoint_index.json)、[parameter summary](../../PARC2026_final/runs/T2EXP-005-xvla-libero-improved/production-two-phase-20260904-r2/parameter_summary.json)。

**反映**：Soft Prompt単独は今回の新しい比較条件。過去LoRA構成を比較候補に追加する。過去batch4稼働だけで今回のLeRobot版・16 GB条件・5 Hz推論を保証しない。学習step_timeは推論latencyではない。

## 10. 学習完了と性能採択は別（F-10）

**実行記録/台帳**：improvedの`best_eval=null`、selection_metric=rollout_success_rate。experiment.jsonのprimary_valueもnull、decision pending。今回確認した証拠から、新8タスク80%達成やこのcheckpoint採用済みとは言えない。planned台帳と学習済み成果物が併存するためstatusだけの判定は不十分。

根拠：[checkpoint index](../../PARC2026_final/runs/T2EXP-005-xvla-libero-improved/production-two-phase-20260904-r2/checkpoint_index.json)、[experiment台帳](../../PARC2026_final/experiments/T2EXP-005-xvla-libero-improved/experiment.json)。

**反映**：既存checkpointは比較用候補。全PARC評価を網羅して不存在と証明したものではなく、性能主張には当該bundleの監査済み評価JSONを追加取得する。

## 11. 推論adapterは再利用前の監査が必要（F-11）

**コード**：既存adapterはabsolute targetを位置0.05 m・回転0.5 radで7D deltaへ変換し、[-1,1]にclip、gripperは0.5閾値で±1へ変換する。numpyの小さな変換器は再利用候補だが、退化回転やNaNを完全拒否する契約とは異なる。推論はtarget dequeを持つ同期方式で、今回の非同期ROS scheduler/期限/UUID/watchdogを実装済みとは言えない。

根拠：[action adapter](../../PARC2026_final/experiments/T2EXP-005-xvla-libero-improved/code/common/action_adapter.py)、[policy](../../PARC2026_final/experiments/T2EXP-005-xvla-libero-improved/code/inference/policy_impl.py)、[単体試験](../../PARC2026_final/experiments/T2EXP-005-xvla-libero-improved/tests/test_action_adapter.py)。

**反映**：PARC互換clipはsim decoderのversionに明示する。real経路の安全拒否と混同せず、過去関数をそのままRT制御に流用しない。

## 12. 評価と環境分離の知見（F-12）

**運用契約**：PARCでは初期stateはseed抽選ではなくoffset+episodeで選択される。paired比較はseedだけでなく初期state hash・task・評価器/runtime hashを照合し、選択/確認で初期state再利用を避ける。成功とcollisionを別記録し、PARCのcollisionは「非操作対象物の最大L1変位>1 mm」という競技固有定義を持つ。

根拠：[AGENTS](../../PARC2026_final/AGENTS.md)、[評価契約](../../PARC2026_final/docs/EVALUATION_CONTRACT.md)、[環境契約](../../PARC2026_final/docs/ENVIRONMENTS.md)。

**反映**：新計画でも初期state hashと非重複を必須化する。PARCのcollision判定・Python3.10 Docker・LeRobot0.4.4という条件を新実機計画へ無条件に強制しない。学習から推論へのexport境界とgolden testの考え方を再利用する。

## 13. 現在の資産有無と実機実績（F-13）

**今回ファイル確認**：`PARC2026_final/artifacts/models/xvla-libero/base`は存在しない。旧preの`experiments/XVLA_libero/base_model/X-VLA-Libero`にはconfig・tokenizer・sourceが残るが、それだけで重み一式ありとは判定しない。今回参照したrunのadapterファイルは残存する。独立ロードとbase weight hash確認は未実施。

**未確認**：今回の照合資料にPanda実機、FCI、ROS 2非同期推論、150 ms推論達成を裏付ける実行証拠は確認していない。これらをPARCの成功実績として記載しない。

**反映**：D-02に重み復元・bundle完全性を追加、D-06〜D-09はOPENを維持する。

## 今回の確認試験

PARC `.venv/bin/python -m pytest -q`で以下の既存試験を実行し、**4 passed**（2026-10-30）。

- `experiments/T2-Xvla-libero/tests/test_t2_xvla_layout.py`
- `experiments/T2EXP-005-xvla-libero-improved/tests/test_action_adapter.py`

これは限定fixtureの回帰確認であり、教師全軌道再生、モデルロード、GPU、Docker固定評価、ROS 2/実機の検証ではない。文書間リンク・原案hash・参照ファイルhashを別途検査し、`PARC_EVIDENCE.json`に証拠snapshotを記録する。

## 追加照合：予選資料で確認できた成功実績（F-14〜F-17）

追記日：2026-10-30。初回照合は本選側に偏っていた。以下はそれを補完する予選の証拠であり、「PARCには学習後の成功評価が確認できない」という全体への解釈を訂正する。F-10は本選improved実験についての記述としてのみ有効。

### F-14：採用した学習法と提出物

[予選解法報告](../../PARC2026/PARC2026_pre/report.md)に、2toINF/X-VLA-Liberoから公開4taskの1,595episodeを用い、LoRA rank8/alpha16/all-linear＋Soft Prompt/Action Encoder/Decoder保存、batch2、AdamW、LR1e-4、weight decay0、betas(0.9,0.95)、grad norm1、freeze1000、warmup2000、cosine無効、100,000stepで学習し、**body-frame版`submission_ckpt100000_bodyframe.zip`を最終採用**したと記録されている。16 GBでは大batch/full FTでOOMがありbatch2を使用したと報告される。

報告されたbase重みhashは`260cc58869125b826e93bcaa60bca3ea37bcc7aaf893d368a991ca14fde9f0c8`、adapter hashは`fd6ae340d2176f005dc3eb237e14eacdc859493b637f1c50afdd9d2eef825c0c`、提出ZIP hashは`e9c49c5733867422d0c204679d769c2b571ef8c21cbfefe6f277dfbde385165f`。これは過去記録の値であり、今回base/ZIP実体の再hashやロードを行ったものではない。

### F-15：採用body-frame版のローカル評価は20/20成功

[保存されたDocker body-frame 100k評価JSON](../../PARC2026/PARC2026_pre/experiments/XVLA_libero/artifacts/local_eval/docker-body-100k-5ep/server_8000.json)には、track1/libero_t1、4task、各5episode、計20episode、mean_success_rate=1.0、各task success_rate=1.0、mean_collision_rate=0.0が記録されている。timestampは2026-08-08T01:32:29.364190+00:00、seed42、max steps300。ファイル名のDocker表記と報告を根拠に呼称しており、今回Docker実行やruntime完全再現を行ったわけではない。

**結論**：予選には追加学習後の成功評価の保存実績がある。「学習完了だけ」との先の説明は予選まで含めると不正確だった。20試行のローカルsim結果であり、公式リーダーボードの得点、独立holdoutでの一般化、新8task、実機70%の証明ではない。

### F-16：35%診断と100%評価は変換経路が異なる

[診断書](../../PARC2026/PARC2026_pre/experiments/XVLA_RL/reports/lora_100k_diagnosis/REPORT.md)と[paired JSON](../../PARC2026/PARC2026_pre/experiments/XVLA_RL/logs/base_vs_lora_t1_10ep.json)には、controller-frame経路の4task各10episodeでBase 37/40=92.5%、LoRA100k 14/40=35%というtask completionが残る。同JSONのcollision-freeはBase34/40=85%、LoRA13/40=32.5%であり、成功定義を混ぜない。

診断はactionの1stepずれ、過大更新、scene variant混在を原因候補とし、同時刻state/actionと低LR LoRA-onlyを推奨する。一方、予選最終報告は**学習targetと整合するbody-frame**を採用している。どちらかを無視して「LoRAは必ず失敗」「controller-frameが常に正解」と結論しない。frame、target生成、時刻alignment、checkpoint hashをセットで復元し、同一protocolで差を再検証する必要がある。

### F-17：Base凍結＋gated residualにも改善記録

[Residual手法](../../PARC2026/PARC2026_pre/experiments/XVLA_base_improve/README.md)はX-VLA/Soft Prompt/Action層を固定し、小さなgated residualを7D deltaへ加える追加学習。別手法として扱う。

[残存paired JSON](../../PARC2026/PARC2026_pre/experiments/XVLA_base_improve/logs/residual_r2_official_10ep.json)をrecordsから再集計すると、4task各10回でBase task completion35/40=87.5%、collision-free32/40=80%、Residual completion38/40=95%、collision-free35/40=87.5%。[heldout visual JSON](../../PARC2026/PARC2026_pre/experiments/XVLA_base_improve/logs/residual_r2_heldout_visual_10ep.json)にもcollision-free 87.5%→90%という保存集計がある。ファイル名のofficialは公式提出得点を証明しない。

本手法の原案8task/実機への転用は未検証。今回の初期baselineはF-14/15の2toINF LoRA構成とし、残差学習は必要時の比較候補とする。LoRAの100k固定継続は必須にせず、固定rollout validationでcheckpointを選ぶ。

### 今回への必須反映

- PARC-09：予選再現baselineはbatch2のLoRA＋Soft Prompt＋Action層構成とbody-frame提出adapterを一組で調査する。Soft Prompt単独を「予選実績の再現」と呼ばない。
- PARC-10：F-15とF-16を分け、同一checkpointでframe/前処理/時間整合のpaired ablationを行う。実測と整合するまで新8task学習や実機指令へ流さない。
- PARC-11：予選の1e-4/100kを無条件の最適設定とせず、Base・予選再現・低LR候補を開発評価で比較する。最終実機70%は独立に検証する。
