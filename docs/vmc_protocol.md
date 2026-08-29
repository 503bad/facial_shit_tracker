# VMC Protocol 送信仕様メモ（Marionette宛て）

出典: https://protocol.vmc.info/specification / https://protocol.vmc.info/marionette-spec

## ポート
- Marionette（VSeeFace等の受信側）は **UDP 39539** で待受 → 本アプリはここへ送信
- Performer（ばもきゃ本体等）は UDP 39540 で待受（生トラッカーを流し込む場合のみ）
- OSC over UDP、片方向・ハンドシェイクなし

## メッセージ（アドレス / OSC型タグ / 引数）
- `/VMC/Ext/OK` `iiii` → loaded(0/1), calibration_state(3=Calibrated), calibration_mode(0), tracking(1/0)
  - 定型: `1 3 0 1`
- `/VMC/Ext/T` `f` → アプリ起動からの相対秒。毎フレーム送る（keep-alive）
- `/VMC/Ext/Root/Pos` `sfffffff` → `"root"`, px,py,pz, qx,qy,qz,qw（v2.1形式は末尾に scale×3, offset×3 = `1,1,1,0,0,0`）
- `/VMC/Ext/Bone/Pos` `sfffffff` → ボーン名, ローカル位置xyz, ローカル回転 **クォータニオンは (x,y,z,w) 順**
  - 位置・回転とも**親ボーン相対のローカル値**。位置はバインドポーズのローカルオフセット固定で回転のみ動かすのが実用形
- `/VMC/Ext/Blend/Val` `sf` → 名前, 値0.0–1.0（大文字小文字区別）
- `/VMC/Ext/Blend/Apply` 引数なし → Valの一括反映。フレーム末尾に1回

## 座標系
- Unity準拠: 左手系, Y-up, Z前方, **メートル**単位

## ボーン名（Unity HumanBodyBones、そのままの文字列）
Hips, Spine, Chest, UpperChest, Neck, Head,
LeftShoulder, LeftUpperArm, LeftLowerArm, LeftHand,
RightShoulder, RightUpperArm, RightLowerArm, RightHand,
LeftUpperLeg, LeftLowerLeg, LeftFoot, LeftToes,
RightUpperLeg, RightLowerLeg, RightFoot, RightToes,
Jaw, LeftEye, RightEye,
{Left,Right}{Thumb,Index,Middle,Ring,Little}{Proximal,Intermediate,Distal}
（親指も Proximal/Intermediate/Distal。小指は Little*。全55）

## 送信レート・バンドル
- 30〜60Hzが標準。1UDPパケット≦1500バイトでOSCバンドル化（55ボーンは1パケットに収まらないため分割、例: 20ボーン/バンドル）
- フレーム内順序: OK/T → Root → Bone... → Blend/Val... → Blend/Apply（Applyが最後）

## VRM 0.x / 1.0 ブレンドシェイプ名
- 0.x: Joy/Angry/Sorrow/Fun, A/I/U/E/O, Blink_L/Blink_R（TitleCase）
- 1.0: happy/angry/sad/relaxed, aa/ih/ou/ee/oh, blinkLeft/blinkRight（小文字）
- VSeeFaceは0.x系。※本アプリでは表情はiFacialMocap側で送るのでVMC側Blendは基本未使用
