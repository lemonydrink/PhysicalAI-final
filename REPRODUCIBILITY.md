# 再現手順・必要資産

このリポジトリは実施時点の作業コードと監査可能な中間結果を保存する。SFTはまだ完了していない。

## 環境

Linux x86_64、CUDA対応NVIDIA GPU、Python3.12、uvを想定する。シミュレーターと学習系はNumPy/Transformersの依存が異なるためvenvを分ける。

```bash
cd workspace
uv sync --locked --extra ml --extra sim --extra dev
uv sync --locked --project backends/smolvla
.venv/bin/python -m pytest tests -q
backends/smolvla/.venv/bin/python -m pytest backends/smolvla/tests -q \
  -k "not prepared_smoke and not libero_initializer and not burgq_uses and not raw_conversion"
```

全backendテストを実行するには、以下のモデルに加えsmoke HDF5・rawデータfixtureも準備し、上記の `-k` 指定を外す。公開コピーの構文・資産不要テストの実行結果はreportsに保存する。

## 別途取得するもの

1. BurgQモデルの固定revision：`238fb0654cd598e287f8b6c2d64bd09af239c76f`。重みSHA256は実験レポートを参照。元のconfig・processor・normalizerも保存する。
2. SmolVLM2-500M-Video-Instructのconfig/tokenizer等、revision `7b375e1b73b11138ff12fe22c8f2822d8fe03467`。既存コードはローカルのVLM参照先を使用する。
3. Sylvest/libero_plus_lerobot、revision `22c57433fef692b5b9ecc0795344daac7fa867a5`。`workspace/scripts/download_eight_dataset.py --help` と `workspace/docs/DATASET_ACQUISITION.md` を参照。
4. LIBERO-plusのシーンassets。`workspace/third_party/libero/libero/libero/assets` に配置またはsymlinkする。出所は `workspace/docs/LIBERO_SOURCE.json` を参照。この外部symlinkは公開物から除いた。

LeRobotの旧config読み込み互換性のため、BurgQ configの `pretrained_revision: null` だけを互換コピーから除いた。重みは変更しない。`compatible/provenance.json` は元ファイルと変換内容のhashを記録する。現行ローダーはこのprovenanceを検査するため、重みだけ置いても起動しない。

## データ・設定の再構築

`reports/2026-11-01-burgq/manifests/` は元取得情報、評価用旧split、監査後のgrouped-v2 splitを記録する。画像・actionそのものは同梱しない。

raw取得→`prepare-raw`→元軌道の操作列hash単位でsplit→manifest作成→入力監査の順で行う。学習splitは365,887フレーム。監査時のgroup作成スクリプトは `reports/2026-11-01-burgq/audit/source/` に保存しているが、元環境のパスや比較用元データの準備が必要であり、そのまま実行する一般CLIではない。

別ディレクトリではmanifest内のHDF5パス、plan内のconfig・train/eval manifest・predecessor・監査のパス、必要なローカルモデル参照先を再設定する。移設後はconfig/manifest hashが変わるため監査を実施し直す。公開済みpassed/hashを新しいデータに流用して検査を迂回しない。

旧splitはbaseline protocolの記録維持のため評価manifestとして固定した。評価は学習HDF5窓を読まず、モデル付属normalizerを使う。新splitはSFTのみで使う。

## 起動の入口

```bash
cd workspace
export SMOLVLA_CONFIG="$PWD/backends/smolvla/configs/burgq-staged-batch32.json"
export SMOLVLA_STATE_ENCODING=libero_quat2axisangle
export WANDB_PROJECT=physicalai-smolvla-eight
export WANDB_MODE=offline
backends/smolvla/.venv/bin/python backends/smolvla/staged_job.py --help
backends/smolvla/.venv/bin/python backends/smolvla/staged_training.py --help
```

準備・再監査済みのjobを指定して `staged_job.py --job <job>` を起動すると、simulator起動→baseline80→SFT→2,000更新ごとのrollout→best選択を行う。保存済みplanは実験証跡で、コピーするだけで別PCに予約サービスが登録されるわけではない。

W&Bのオンライン利用時は `wandb login` を実行し、`WANDB_PROJECT`・`WANDB_ENTITY` を設定する。学習CLIは `--wandb-mode online|offline|disabled` に対応する。記録されたjob wrapperはoffline固定のため、オンライン化する場合はwrapperの環境設定とCLI指定も変更する。APIキーを設定やコードへ書き込まない。

ROS 2メッセージ定義とbridgeスクリプトは `workspace/ros2_ws/src/` と `workspace/scripts/`。ROS 2実行環境自体は外部のPixi環境に依存し、今回のSmolVLA測定には使っていない。旧X-VLA向けbootstrapは旧モデルを取得するため、BurgQ評価の手順と混同しない。

## 公開範囲

コード、設定、lock、設計資料、第三者ソースの既存snapshot、監査、全80trial JSON、baseline成功動画各タスク1本、途中学習metricsを含む。重み、raw/HDF5学習データ、外部scene assets、venv、build、W&Bバイナリ、旧大容量分析archive・第三者のデモ動画とテスト用tensor fixtureは除外した。元の実行フォルダと学習プロセスはそのままローカルに残す。


## 最終ROS 2評価（追加）

`ros2/pixi.toml` と `ros2/pixi.lock` にROS 2 Jazzy環境を追加保存した。Pixi自体を導入してから `pixi install --locked --manifest-path ros2/pixi.toml` を実行する。学習venvをdeactivateした別のbashで `source ros2.env`、`cd workspace`、`bash scripts/build_ros.sh` を実行してcustom messageをbuildする。

`workspace/scripts/run_smolvla_ros.sh` はこのROS環境とML環境を分ける。`PHYSICALAI_ROS_ENV` で別のROS環境設定を指定できる。公開 `.service.example` は元ホストのパスを証跡として保持するため、別ホストへそのまま登録せず配置先を修正する。

`backends/smolvla/final_ros2.py --job <audited-job> --policy-smoke` が2並列の実環境DDS配線と短いpolicy rolloutを確認し、`--wait` が学習・モデル選択とプロセス退出後の最終80回を予約する。最終評価はindex20〜29、2並列、best checkpoint、ROS 2 DDS。中間評価は4並列HTTPを維持する。

実機は未検証。vendor driver、実測カメラ/座標/制御gainの校正、独立watchdog・非常停止・controller commissioningが必要。詳細は `workspace/docs/FINAL_ROS2_AND_HARDWARE.md` を参照。デフォルトの予約はsimulationのみで実機を有効化しない。
