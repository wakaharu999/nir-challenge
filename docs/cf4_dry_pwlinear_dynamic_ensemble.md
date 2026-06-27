# cf×dry 動的アンサンブル: 全MCモデル × 高含水専用モデル (2026-06-27)

中低MCに強い **cf**([collinear_filtered_ensemble.md](collinear_filtered_ensemble.md) 系を更新)と、高含水外挿に強い
**dry**([dry_band_gauss_pwlinear_model.md](dry_band_gauss_pwlinear_model.md) の採用モデル)の **推論値を推定MC依存の
動的シグモイドゲートで結合**して LOSO 検証した記録。結合は予測値レベルのみ、特徴量は各モデル定義のまま。
実装/再現: `notebooks/cf4_dry_pwlinear_dynamic_ensemble.ipynb`。共通入力 `train_clean.csv`(1270行, 13樹種, MC 0.8–298.6%)。

## モデル定義

- **cf(全MC担当)**: 共線形フィルタ20特徴(候補72=9バンド×8統計量 → 共線形除去|r|≥0.90 → 含水率相関|r|>0.30
  → 手動除外3本、**fold内選択=リーク無し**)× **ExtraTrees / LightGBM / SVR(rbf) / MLP の均等平均**。
  MLP は目的標準化(TransformedTargetRegressor)した小型(64)・強正則化で、誤差が独立な多様性源
  ([loso_collinear_filtered_models.ipynb](../notebooks/loso_collinear_filtered_models.ipynb))。
- **dry(高MC専用)**: 乾物 mean+slope混合5特徴(CH1270_mean/CH1270_slope/CH2210_mean/CH2320_mean/CL2360_mean)
  × **区間線形ヒンジ基底 K=8 + Ridge α=1**。裾が線形なので学習範囲外(高MC)を素直に外挿。

## 結合式(動的重み)

推定MC(ゲート信号)が高いほど dry を重視する単調シグモイドゲート:

```
pred = (1 - w)·cf + w·dry,   w = sigmoid((gate_MC - center) / scale)
```

- ゲート信号候補: `cf` / 両者平均 / 両者max。center・scale と合わせ LOSO macro-RMSE 最小でグリッド選択。
- 選定結果: **gate=max, center=160, scale=40**(推定MC≳160% で滑らかに dry へ移行)。
  dry が区間線形(旧 exp/quad より高MC精度↑)になったため、旧記録(center=190, scale=5 の急峻ゲート)より
  **緩やかなゲート**が最適化された。

## LOSO 結果

| モデル | overall-RMSE | macro-RMSE | ベイスギ(最高MC) |
|---|---|---|---|
| cf 単独 | 21.36 | 17.30 | 80.7 |
| dry 単独 | 27.67 | 26.78 | 32.1 |
| 参考: 固定 0.5/0.5 | 20.43 | 19.38 | 52.9 |
| **動的アンサンブル** | **16.94** | **15.65** | **52.9** |

dry 単独の MC>120 RMSE=28.03(120–200%=25.97 / >200%=39.7)。

樹種別RMSE(平均MC昇順、抜粋):

| 樹種 | 平均MC | cf | dry | 動的ens |
|---|---|---|---|---|
| ウエンジ | 15 | 14.9 | 15.5 | 13.6 |
| スプルース | 37 | 7.7 | 23.8 | 10.2 |
| ベイマツ | 42 | 4.7 | 24.2 | 8.2 |
| ナラ | 42 | 8.7 | 15.7 | 5.9 |
| クリ | 73 | 7.8 | 28.6 | 9.3 |
| イチョウ | 78 | 12.9 | 40.8 | 13.5 |
| **ベイスギ** | **145** | **80.7** | 32.1 | **52.9** |

## 所見

- **動的アンサンブルは cf 単独より macro(17.30→15.65)・overall(21.36→16.94)とも大きく改善**。
  最大の改善は高MCのベイスギ(80.7→52.9)で、中低MC樹種はおおむね cf 値を維持(設計通り)。
- 緩やかなゲート(scale=40)が macro 最適。区間線形 dry は高MC精度が高く、ベイスギだけでなく
  ナラ/ウエンジ等の中域でも軽く dry 側へ寄せることが macro に有利に働いた(net で改善)。
  反面スプルース/ベイマツ等はわずかに悪化(7.7→10.2, 4.7→8.2)。固定50:50は中域を崩し劣化(macro 19.38)。
- **乾物特徴は dry モデル内に閉じ込めるのが要点**。全MCモデルへ直接足すと樹種間相関の罠で macro が悪化する
  ([loso_collinear_filtered_models.ipynb](../notebooks/loso_collinear_filtered_models.ipynb) の探索)。役割分担で両取り。
- center=160/scale=40 はグリッド端寄り(scale上限)。さらに緩いゲートで微改善の余地はあるが過適合リスク。

## Optuna 全ハイパラ同時チューニング(2026-06-27)

全18ハイパラ(cf 特徴しきい値 corr_th/mc_th、ExtraTrees/LightGBM/SVR/MLP の各パラメータ、cf 4モデル重み、
dry K/α、ゲート信号/center/scale)を **単一 Optuna study(TPE多変量, 120 trial)で同時最適化**(目的=LOSO macro)。
再現: `notebooks/cf4_dry_optuna_tuning.py`、最良値 `notebooks/cf4_dry_optuna_best.json`。

| 指標 | 既定(本doc) | Optuna 最良 |
|---|---|---|
| macro-RMSE | 15.65 | **14.47** |
| overall-RMSE | 16.94 | **16.22** |
| ベイスギ | 52.9 | 52.95 |

- 最良: corr_th=0.967, mc_th=0.360 / ET(n=800,leaf=2) / LGB(n=300,lr=0.041,leaves=40) /
  SVR(C=3.16,ε=0.011) / MLP(h=48,α=0.243,lr=0.0019) / **cf重み ET:LGB:SVR:MLP=1:1.03:0.96:1.93** /
  dry K=8,α=1.15 / **gate=max, center=172, scale=37**。
- param importance 首位は **gate信号の選択(0.36)** で `gate=max` 必須。改善の主因は **cf 重みで MLP を約2倍**に
  厚くしたこと(独立な多様性源を増強)。dry K=8・ゲート center/scale は既定近傍に収束=結合式は頑健。

## ネストLOSO 検証(honest 推定, 2026-06-27)

楽観バイアスを除くため、**外側LOSO(1樹種除外)ごとに残り12樹種の内側LOSOで上記18ハイパラを Optuna 再選択
→ 除外樹種を予測**(各fold 20 trial、全データ最良値を warm-start)。再現: `notebooks/cf4_dry_nested_loso.py`、
結果 `notebooks/cf4_dry_nested_loso_result.json`。

| 指標 | Optuna(楽観) | **ネストLOSO(honest)** |
|---|---|---|
| macro-RMSE | 14.47 | **17.92** |
| overall-RMSE | 16.22 | **22.74** |
| ベイスギ | 52.95 | **88.6** |

- **悪化はほぼ全てベイスギ由来**: ベイスギ除く12樹種の macro=**12.03**(楽観値とほぼ同等)=中低MC樹種は過適合していない。
- 原因はゲート: ベイスギを除くと内側は全て **MC≤78%** → dryへ切替える必要が無く `center≈224, scale≈5`(=ほぼ切替えない
  急峻ゲート)が選ばれ、外挿段でベイスギ(145%)を cf 天井打ちのまま渡せず大崩れ(88.6)。他樹種は center≈159–190 で安定。
- **結論**: doc記載の「軽い楽観」の正体は『ゲートの center/scale がベイスギの存在を前提に選ばれている』こと。ベイスギは
  学習データ唯一の超高MC樹種ゆえ、それを除くと最適ゲートが原理的に決まらない=**過適合ではなくデータ被覆の限界**。
  実運用(全データでゲート固定)ではベイスギ域も 52.9 で機能する。ただし**未知の超高MC樹種への真の汎化は現データでは保証外**。

## 低含水帯バイアスと log 局所混合(2026-06-27)

現行モデルは **低含水(MC≤50%)を系統的に過大評価**(平均誤差 +4.98、過大率72%)。特に **0–20% 帯で +8.0・過大率95%**
(20–35%=+2.0、35–50%=+1.6 と高MC側ほど解消)。ゲート w≈0 のため **cf(全MC)由来のフロア効果**(学習裾での平均回帰)。
樹種別では平均MCが低い樹種に集中: ヒノキ +13.7・チェリー +25.6・ウエンジ +9.2・スプルース +5.4(いずれもMC≤50で100%過大)。

対策として cf を **log 目的で再学習した cf_log** を低含水だけに混ぜる。`pred=combine((1−w_low)·cf + w_low·cf_log, dry)`、
`w_low = sigmoid((center_low − cf予測MC)/scale_low)`。再現は `notebooks/cf4_dry_nested_loso.py` の部品を流用、図は
`notebooks/figures/fig3_低含水バイアス診断.png`・`fig4_誤差vsMC.png`・`fig5_低log混合_樹種別予測.png`。

| 構成 | macro | overall | 備考 |
|---|---|---|---|
| 現行(混合なし) | 14.47 | 16.22 | — |
| cf 全域 log | 15.98 | 17.85 | 20–50%帯の数樹種が崩れ悪化(ホワイトオーク−14.2/イチョウ+15.3/クリ−11.4) |
| **低MCゲート log 局所混合** | **14.36** | **16.09** | center_low=20, scale_low=2.5 |

- **log は cf 全域に掛けると悪化、低含水だけ局所混合すれば net 微益**(「低含水log・高含水線形」の役割分担が正解)。
- 閾値は細密グリッド(center_low 1刻み×scale_low 0.5刻み)で **center_low≈20, scale_low≈2.5** に明確な谷(粗グリッドの20/3と一致、
  真MC<20%で過大が顕在化する境界と物理的に整合)。ただし谷は浅く改善は **+0.11 と小幅**。
- 効果は限定的: 0–20%帯の過大 +8.0→+6.2、ヒノキ RMSE 13.9→11.7・ウエンジ 10.6→8.5 が改善。中高MC樹種は w_low≈0 で現行と一致(無害)。
- **未解決**: チェリー(+25.6)は log でも不変=フロア効果でなく系統誤差の別要因。ホワイトオークは僅かに過小方向へ悪化。
- center_low/scale_low も全LOSO集計で選んでおり楽観バイアス込み(谷が浅いためネストLOSOで消える可能性)。採用前に厳密検証を推奨。

## リーク注意

- cf の特徴選択・dry のバンド選定・本結合の center/scale はいずれも全fold集計で固定 = **軽い楽観バイアス**。
  厳密な外挿見積りには fold 内で全選択を行うネストLOSO が必要([loso-validation のネスト化])。
- ノット・標準化・特徴選択はすべて各 fold の学習側のみで fit 済み(モデル内部のリークは無し)。
- 共通入力 `train_clean.csv`(ベイスギ外れ値除去後)= dry を良好な形で使うため(全 `train.csv` だと dry のベイスギが悪化)。

## 再現

`notebooks/cf4_dry_pwlinear_dynamic_ensemble.ipynb`:
- cf特徴セル → cf 4モデル OOF(均等平均)、dry 区間線形 OOF、動的重みグリッド探索、
  「macro比較 + 動的重み曲線 + 樹種別RMSE」図、「樹種別 真値 vs LOSO予測(横軸=樹種内サンプル番号, MC昇順)」図。
- データ Shift-JIS、メタ列 `sample number, species number, 樹種, 含水率`、以降が波数(cm⁻¹)降順の吸光度。
