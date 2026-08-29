# iFacialMocap v1 送信仕様メモ

出典: https://www.ifacialmocap.com/for-developer/ ＋ VMagicMirror受信実装 ＋ ExpressionAppBridge送信実装（動作実績）

## トランスポート
- UDP、ASCIIテキスト1行=1フレーム、約60FPS
- 送信先: **PCのIP:49983**（受信アプリが49983で待受）
- ハンドシェイク: PC側が `iFacialMocap_sahuasouryya9218sauhuiayeta91555dy3719` をiPhone:49983へ送る
  （`|sendDataVersion=v2` 付きならv2要求）。**ハンドシェイク無しの直接送信でも受信される**
  （VSeeFace/VMagicMirrorで実績）。本アプリは49983でハンドシェイクを待ちつつ、設定先へ直接送信。

## v1ペイロード
```
<name>-<int0..100>|...|=head#rx,ry,rz,px,py,pz|rightEye#rx,ry,rz|leftEye#rx,ry,rz|
```
- ブレンドシェイプ値は**整数0〜100**（`-`区切り。小数・負数不可 — int.TryParseで捨てられる）
- `=` はパケット全体で**ちょうど1個**（`head#`の直前）
- **末尾の `|` は必須**
- head回転はEuler度（Unityの `Quaternion.Euler(rx,ry,rz)` にそのまま入る）、位置はメートル
- 目はEuler度3値（多くの受信側はeyeLook*ブレンドシェイプを優先するが両方送る）
- 前フレームとバイト同一のパケットはトラッキングロス扱いされる（VMagicMirror）→ 値が完全静止しないよう注意
- v2との違いは区切りが `-`→`&` のみ

## 52キー（_L/_Rサフィックス形式・camelCase）
browInnerUp, browDown_L/R, browOuterUp_L/R,
eyeLookUp_L/R, eyeLookDown_L/R, eyeLookIn_L/R, eyeLookOut_L/R,
eyeBlink_L/R, eyeSquint_L/R, eyeWide_L/R,
cheekPuff, cheekSquint_L/R, noseSneer_L/R,
jawOpen, jawForward, jawLeft, jawRight,
mouthFunnel, mouthPucker, mouthLeft, mouthRight,
mouthRollUpper, mouthRollLower, mouthShrugUpper, mouthShrugLower, mouthClose,
mouthSmile_L/R, mouthFrown_L/R, mouthDimple_L/R, mouthUpperUp_L/R,
mouthLowerDown_L/R, mouthPress_L/R, mouthStretch_L/R, tongueOut

## NVIDIA 53係数からの変換
- browInnerUp = avg(browInnerUp_L, browInnerUp_R)
- cheekPuff = avg(cheekPuff_L, cheekPuff_R)（EnableCheekPuff=1時のみ有効値）
- tongueOut = 0（NVIDIAは非対応）
- 残り50キーは同名（NVIDIAも_L/_R形式）
