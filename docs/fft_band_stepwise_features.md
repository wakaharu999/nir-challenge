# FFT弱マスク由来バンドからの統計特徴 + stepwise探索 (2026-06-26)

water_band / fft_denoise / inverse_fft の各 notebook で、含水率とのグラデーションが
目視確認できたバンド群から統計特徴を作り、forward stepwise で LOSO macro-RMSE を
最小化する組合せを探索した記録。前パイプライン([snv_epo_strength_features.md](snv_epo_strength_features.md))とは別系統の試行。

## 前処理

- **ベーススペクトル**
  - 元スペクトル: 生波形そのまま (nm昇順)
  - 弱マスク: 逆FFTで中心±15%の点だけ残す平滑化 `fft_denoise(X, keep_frac=0.15)` 後の波形
- **SNV**: 行単位 (`nir_features.snv`)。`snv_*`=SNV後にSG / `raw_*`=生波形に直接SG。
- **SG**: `savgol_filter(win=11, poly=2)`、SG1=1次微分・SG2=2次微分。
- FFT/SNV/SG/統計量は全て**行単位** → 特徴行列はリーク無しで全体一括計算可。モデル(RF)のみ fold 毎学習。

## 候補特徴 = 9バンド × 8統計量 = 72本

バンド (前処理 × 帯域[nm]、想定意味):

| 名前 | ベース | 前処理 | 帯域[nm] | 想定 |
|---|---|---|---|---|
| snv_1430_1470 | 元 | SNV | 1430–1470 | 吸光度 |
| snv_sg1_2000_2100 | 元 | SNV→SG1 | 2000–2100 | 強度(尖り) |
| snv_sg1_2400_2480 | 元 | SNV→SG1 | 2400–2480 | 強度(尖り) |
| snv_sg1_1650_1670 | 弱マスク | SNV→SG1 | 1650–1670 | 強度 |
| snv_sg2_1910_1940 | 弱マスク | SNV→SG2 | 1910–1940 | 強度 |
| snv_sg1_2020_2100 | 弱マスク | SNV→SG1 | 2020–2100 | 強度 |
| raw_1050_1300 | 弱マスク | 生 | 1050–1300 | 吸光度 |
| raw_sg2_1430_1470 | 弱マスク | 生→SG2 | 1430–1470 | 振幅 |
| raw_sg2_1600_1700 | 弱マスク | 生→SG2 | 1600–1700 | 振幅 |

統計量 (バンド内ベクトル B、波長 w から行単位): `mean, std, min, max,
amp(=max−min), absmax(=max|B|), slope(1次回帰傾き), cr_depth(両端直線からの最大へこみ)`。
命名は `バンド名|統計量`。

## 検証

- LOSO (Leave-One-Species-Out, 13樹種)、指標は **macro RMSE**(樹種等重み)。
- モデル RandomForest。探索は n_estimators=150、確定後の最終評価は 400。
- forward stepwise: 改善幅 <0.05 で停止、最大10本。

## 結果

| step | 特徴 | macroRMSE |
|---|---|---|
| 1 | snv_sg1_2000_2100\|max | 25.26 |
| 2 | snv_sg1_2020_2100\|cr_depth | 20.67 |
| 3 | raw_sg2_1430_1470\|min | 18.32 |
| 4 | snv_1430_1470\|slope | 17.30 |
| 5 | snv_sg1_2000_2100\|cr_depth | 17.18 |

- **最終 5特徴で LOSO macro-RMSE = 17.19** (n_est=400)。
- 全72特徴投入は **18.82** と悪化(絞った方が良い)。
- 樹種別: ベイスギ 59.3 が突出してmacroを支配、次いでイチョウ 23.7・チェリー 23.3。
  良好なのはトチ 8.4・ナラ 8.6・ベイマツ 9.8。

## 重要な注意 (リーク)

この 17.2 は **統計量の選択を全train(=LOSO全fold)を見て固定した楽観値**。stepwiseの
選択判断自体に検証foldが混入しており、[snv_epo_strength_features.md](snv_epo_strength_features.md)
系で確認した「全trainで固定=22.7 / fold毎再抽出=27.8」と同じ構図。**実際の外挿見積りは
これより悪化する**。正直な数字には fold 内で stepwise をやり直すネストLOSOが必要。

ただし FFT弱マスク由来のEDAで選んだこのバンド群は、楽観値ですら前パイプラインの
リーク無し見積り(約28)を大きく下回っており、素性は有望。

## 再現

スクリプト: `scratchpad/stepwise_loso.py`(本リポジトリ外の作業用)。
`BANDS`(帯域)と `stats()`(統計量)を編集して再探索可能。
