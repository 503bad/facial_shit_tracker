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
  stereo/           2カメラ奥行き検出（ベータ・独立ON/OFF、下記参照）
    calibration.py    画角ベース近似校正・対応点からの相対姿勢推定・三角測量
    capture.py        カメラB用タイムスタンプ付きキャプチャ
    mp2d.py           カメラ別MediaPipe観測（構造化2D＋world）
    fusion.py         関節ごとの状態機械（ステレオ/単眼/予測/復帰/喪失）
    engine.py         時刻整合ペアリング・融合・既存契約への結果アダプター
    ui.py             設定パネル・クリック校正ダイアログ
    replay.py         観測ログの再生・検証ツール
docs/               調査したプロトコル仕様メモ
tests/              合成データによる自動検証（test_stereo_synthetic.py）
```

## 2カメラ奥行き検出（ベータ・オプション機能）
2台目のウェブカメラを使い、体の各関節を三角測量して奥行き（前後方向）を安定して
推定するオプション機能です。**初期値はOFF**で、OFFの間は追加の処理・カメラ取得は
一切動かず、従来どおりの1カメラ動作です。ONでも初期化に失敗した場合（カメラB
なし・校正なし等）は自動的に従来の1カメラ動作へ戻ります。

**セットアップ**
1. 2台のカメラを同じ高さ・間隔約30cm・ほぼ平行な向きで、剛性のある台に固定
   （印刷マーカーやキャリブレーションボードは不要）
2. GUI右側「2カメラ奥行き検出（ベータ）」でカメラB・画角（カメラ仕様の値）・
   基線長（レンズ中心間の実測距離）を設定
3. トラッキング停止中に「校正...」→ 静止画を撮影 → 左右の画像で同じ物理的な角
   （机・棚・家具など）を 左→右 の順に 20〜30組 クリック（画面全体と手前・奥に
   分散させる）→「校正を計算」→「保存」
   - 検証モードで対応点をクリックすると3D距離が表示されるので、メジャー等の
     実測と比較して精度を確認できます
4. 「2カメラ追跡を有効にする」をON（トラッキング中でも切替可能）

**動作**
- 両カメラで見えている関節はステレオ（三角測量）、片方だけの関節は単眼＋奥行き
  保持、見えない関節は短時間の予測→無効、と関節ごとに状態管理します
- 検出が復帰した直後の観測は数フレームの整合確認を経てから滑らかに反映される
  ため、復帰時の飛び・暴れを抑えます
- 「前後移動を送信」ONで、アバターのルート（Hips）が実際の前後移動に追従します
  （従来モードでは前後は動きません）
- ステータス行にペア率・時刻差・立体化関節数などを表示。プレビュー右下に
  カメラBの小窓を表示できます
- カメラを動かした・撮影設定を変えた場合は校正をやり直してください

設定は `stereo_settings.json`・校正は `stereo_calibration.json` に保存され、
既存の `settings.json` には一切書き込みません。「観測ログを記録」ONで
`stereo_logs/` にJSONLの観測ログが保存され、
`python -m mocap_studio.stereo.replay <ログファイル>` で同じ入力から融合処理を
再現・検証できます（映像は保存しません）。

制限: MediaPipeバックエンド専用（NVIDIAバックエンドでは無効）。対象は1人。
校正は公称画角にもとづく近似校正のため、絶対距離には数%の誤差が残り得ます
（送信されるのはニュートラル相対の移動量なので実用上の影響は小さい）。

## 平滑化について
強度0%はフィルタ無効（生値）、100%は最強。表情・頭・体は One Euro /
slerpの適応フィルタなので、強くしても速い動きの遅延は比較的小さいままです。

## 配布パッケージの作成（開発者向け）
`make_release.bat` をダブルクリックすると `release\MocapStudio-v<version>\` に
配布に必要なファイルだけ（`mocap_studio/`＋同梱モデル、`docs/`、各 bat、
`requirements.txt`、`README.md`）を集め、同名の ZIP も生成します。
`.git`・`.venv`・`settings.json`・VRM・`__pycache__` は含まれません。
バージョンは `mocap_studio/__init__.py` の `__version__` から取ります。
