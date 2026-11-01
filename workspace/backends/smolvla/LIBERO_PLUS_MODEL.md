# LIBERO-PLUS学習済みSmolVLAの取得記録

ユーザー指定により、次に使用するモデルとして `lerobot/smolvla_libero_plus` を取得した。

- 固定revision: `7bb70aa5bc92b82c9239142775d3a173103567ff`（過去PARCで使用した版）
- 保存先: `models/libero-plus-7bb70aa5/`
- 重み: `model.safetensors`、906,712,520 bytes、500 tensors
- 重みSHA256: `a67a056ab0761afa5f5efef8edd45065cef4494735231b7eeeb8328e71fcfb72`
- config、train_config、pre/postprocessor設定と正規化重み、公開元の評価設定・結果も保存した。公開元の評価動画は取得対象外。
- 取得した10ファイルすべてのサイズと公開元blob hashを照合した。SHA256とサイズは `provenance.json` に記録した。

今回は取得までを実施。モデル固有の前後処理・観測とactionの対応を検証してから推論・評価に接続する。
既存SFTの1.25%という評価結果は汎用初期モデルから学習した別checkpointの結果であり、この取得モデルの性能ではない。
