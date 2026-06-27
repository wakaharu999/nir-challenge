# 共線形除去＋含水率相関フィルタ特徴 × 重み付きアンサンブル (2026-06-26)

[fft_band_stepwise_features.md](fft_band_stepwise_features.md) の候補72特徴から共線形・相関で
**20特徴**を確定し、8モデルを LOSO比較→上位3モデルを重み付きブレンドした記録。
macro-RMSE最小を狙う本命系（高MC外挿特化の [dry_band_exp_quad_model.md](dry_band_exp_quad_model.md) とは別目的）。
実装/再現: `notebooks/loso_collinear_filtered_models.ipynb`。データは `train.csv`(1322行, 13樹種)。

## 特徴定義（72 → 20、全て行単位=リーク無し）

1. fft_band候補 **72特徴**(9バンド×8統計量)を再構成
2. **共線形除去**: |Pearson r| ≥ 0.90 のペアから含水率|r|が低い方を貪欲除去(確定5特徴は優先保持) → 35
3. **含水率相関フィルタ**: 含水率|r| ≤ 0.30 を切り捨て＋手動除外3本 → **20特徴**

採用20特徴（含水率|r|）:

| 特徴 | \|r\| | 特徴 | \|r\| |
|---|---|---|---|
| snv_sg1_2000_2100\|max | 0.86 | snv_sg2_1910_1940\|slope | 0.62 |
| snv_1430_1470\|min | 0.77 | snv_1430_1470\|slope | 0.59 |
| raw_1050_1300\|min | 0.76 | snv_sg2_1910_1940\|std | 0.59 |
| snv_sg2_1910_1940\|min | 0.72 | raw_sg2_1430_1470\|mean | 0.55 |
| snv_sg1_2020_2100\|absmax | 0.67 | snv_sg1_2000_2100\|min | 0.47 |
| snv_sg1_1650_1670\|max | 0.65 | raw_1050_1300\|std | 0.47 |
| (他: snv_sg1_1650_1670\|amp 0.44, snv_sg2_1910_1940\|cr_depth 0.41, snv_sg1_2000_2100\|cr_depth 0.40, snv_sg1_2000_2100\|absmax 0.39, snv_sg2_1910_1940\|max 0.36, raw_sg2_1600_1700\|mean 0.35, raw_sg2_1600_1700\|max 0.30, raw_sg2_1430_1470\|slope 0.43) | | | |

手動除外: `raw_sg2_1430_1470|cr_depth`, `raw_1050_1300|slope`, `raw_1050_1300|absmax`。

## モデルと LOSO macro-RMSE（昇順）

線形系(PLS/Ridge/Lasso/SVR)はfold毎にStandardScalerを学習側でfit(リーク防止)、木系はスケール不要。

| モデル | macro-RMSE |
|---|---|
| **LightGBM** | **17.64**（単体最良）|
| RandomForest | 18.62 |
| SVR(rbf) | 20.86 |
| SVR(linear) | 21.74 |
| Lasso | 23.09 |
| PLS | 23.45 |
| Ridge | 23.78 |

## リーク対策LOSO再検証(2026-06-27)＋XGBoost追加

ドキュメントが指摘していた **特徴選択リーク** を排除して再検証。
共線形除去・含水率相関フィルタを **各foldのtrainのみで実行**(fold内採用 平均19.8本)、
RF/LightGBM/XGBoost を追加した4モデルで LOSO macro-RMSE を測定。

| モデル(fold内特徴選択) | macro-RMSE |
|---|---|
| **LightGBM** | **17.58**(単体最良) |
| RandomForest | 18.52 |
| XGBoost(調整後) | 18.65 |
| SVR(rbf) | 21.48 |

閾値掃引(corr_th×mc_th)でも現行 `0.90/0.30`(20特徴)が均等重み4モデルで16.92と最良級
(0.90/0.20=16.91とほぼ同値だが特徴数が少なく頑健)→ **特徴定義は変更不要**。
全train固定の旧楽観値16.94は honest検証でも16.92で再現され、選択リークの影響は軽微だった。

## 重み付きアンサンブル(4モデル)

**木3種(RF/LGB/XGB)は誤差相関0.93–0.95で冗長**(混ぜても効果ほぼ無し:RF+LGB+XGB=17.62≒LGB単体)。
改善源は **唯一誤差が独立(相関~0.66–0.71)なSVR(rbf)**。よってXGBを薄く・SVRを厚めに配分。

- 均等重み 1/4 → macro-RMSE **16.915**(リーク無しの頑健な基準)
- ネスト重み選択(inner-LOSO)は17.76と悪化 → **重みの過剰チューニングは有害**、丸い固定重みが安全
- 採用: **w = LightGBM 0.35 / SVR(rbf) 0.30 / RandomForest 0.20 / XGBoost 0.15 → 16.882**
- 単体最良(LightGBM 17.58)から **−0.70 改善**。改善は実質SVRの相補性によるもの。

## SHAP（解釈）

最良単体 LightGBM を全trainで学習し TreeExplainer。寄与上位は水バンド強度系
(`snv_sg1_2000_2100|max` 等)。詳細は notebook の bar/beeswarm 図。

## 注意・位置づけ

- **木中心＝macro最小には強いが高MC外挿は不可**(RF/LGBMは学習範囲外で天井打ち)。ベイスギ等の
  未知高含水樹種の外挿は [dry_band_exp_quad_model.md](dry_band_exp_quad_model.md) の exp/quad加法系が担う。
  役割分担: **中低域はこのアンサンブル / 高域はexp外挿**。
- **リーク注意**: 特徴選択(共線形除去・相関フィルタ)とアンサンブル重みを、いずれも全train/全fold集計で
  固定＝**楽観値**。厳密な外挿見積りには fold内で選択・重み決定するネストLOSOが必要
  ([fft_band_stepwise_features.md](fft_band_stepwise_features.md) と同じ構図)。
- macro-RMSE は樹種等重み。難樹種(ベイスギ)が支配しやすい。

## 再現

`notebooks/loso_collinear_filtered_models.ipynb`:
- 特徴定義セル(`CORR_TH=0.90`, `MC_TH=0.30`)、モデル定義、LOSO、アンサンブル(simplexグリッド)、
  「樹種別 真値 vs LOSO予測」可視化、SHAP。
- データ Shift-JIS、メタ列 `sample number, species number, 樹種, 含水率`、以降が波数(cm⁻¹)降順の吸光度。
