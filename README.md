# Mocap Studio

1つのウェブカメラでフェイシャル（NVIDIA AR SDK / Broadcast品質）と上半身＋指
（MediaPipe または NVIDIA BodyPose）をトラッキングし、

- **顔** → iFacialMocap v1 形式（UDP、デフォルト 49983）— パーフェクトシンク対応
- **体・指** → VMCプロトコル（OSC/UDP、デフォルト 39539）

の2系統・別ポートで送信するモーショントラッカーです。座って使う前提で、
脚はTポーズ固定・上半身と腕・指のみ駆動します。

## 必要環境
- Windows 10/11 + RTX GPU（Turing以降、NVIDIA AR SDK が必要）
- NVIDIA AR SDK（Broadcast用 Redist）インストール済み
  （`C:\Program Files\NVIDIA Corporation\NVIDIA AR SDK`）
- Python 3.12

## セットアップ（配布版はこちら）
1. **NVIDIA AR SDK** をインストール（RTX GPU 必須）
   https://www.nvidia.com/ja-jp/geforce/broadcasting/broadcast-sdk/resources/ の
   "AR SDK" Redistributable
2. **Python 3.10〜3.12** をインストール（"Add python.exe to PATH" にチェック）
   https://www.python.org/downloads/windows/
3. **`setup.bat` をダブルクリック** — 上記2つの有無を確認し（無ければURLを開いて案内）、
   このフォルダ内に仮想環境 `.venv` を作って必要なライブラリを自動インストールし、
   完了後にアプリを起動します。

MediaPipeモデル（`mocap_studio/models/*.task`）は同梱済み。
2回目以降は `MocapStudio.bat` をダブルクリックで起動（`.venv` を使用）。
エラーを確認したいときは `MocapStudio_debug.bat`（コンソール表示）。

開発者向け（仮想環境を使わない場合）:
```
pip install -r requirements.txt
python -m mocap_studio
```

## 使い方
1. カメラ・解像度を選び「▶ トラッキング開始」
2. 受信側（Warudo / VSeeFace 等）で
   - iFacialMocap受信を有効化（このPCのIP、ポート49983）
   - VMC受信を有効化（ポート39539）
3. 真顔で「顔キャリブレーション」を押すと表情のゼロ点を補正
4. 平滑化スライダー（表情・頭・体・指）は動作中でも即反映

## 構成
```
mocap_studio/
  nvar.py           NVIDIA AR SDK ctypesバインディング (FaceExpressions / BodyPose)
  mp_body.py        MediaPipe Holistic バックエンド（体33点＋両手21点）
  face_pipeline.py  53係数→iFM 52キー変換・キャリブレーション・平滑化
  body_pipeline.py  3D点群→VMCローカル回転リターゲット・指関節角
  ifm_sender.py     iFacialMocap v1 UDP送信（ハンドシェイク応答つき）
  vmc_sender.py     VMCプロトコル OSC送信（バンドル分割）
  smoothing.py      One Euroフィルタ＋クォータニオンslerp平滑化
  vrm_loader.py     VRM解析（ボーンオフセット・パーフェクトシンク判定）
  gui.py            PySide6 GUI
docs/               調査したプロトコル仕様メモ
```

## 平滑化について
強度0%はフィルタ無効（生値）、100%は最強。表情・頭・体は One Euro /
slerpの適応フィルタなので、強くしても速い動きの遅延は比較的小さいままです。
