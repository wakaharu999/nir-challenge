# 動的重み付きアンサンブル: 共線形除去20特徴 × 乾物exp/quad加法 (2026-06-27)

[collinear_filtered_ensemble.md](collinear_filtered_ensemble.md)(中低MCに強い **cf**)と
[dry_band_exp_quad_model.md](dry_band_exp_quad_model.md)(高MC外挿に強い **dry**)の
**推論値を推定MC依存の動的重みで結合**して LOSO 検証した記録。特徴量は各モデル定義のまま、結合は予測値レベルのみ。
実装/再現: `notebooks/dynamic_weighted_cf_dry_ensemble.ipynb`。共通入力は `train_clean.csv`(1270行, 13樹種)。

## 結合式（動的重み）

推定MC(ゲート信号)が高いほど dry を重視する単調シグモイドゲート:

```
pred = (1 - w)·cf + w·dry,   w = sigmoid((gate_MC - center) / scale)
```

- ゲート信号候補: `cf` 予測 / 両者平均 / 両者max。center・scale と合わせ LOSO macro-RMSE 最小でグリッド選択。
- 選定結果: **gate=max, center=190, scale=5**（=推定MC≳185%で急峻に dry へ切替）。
- cf/dry の内部重みは各ドキュメントの確定値(cf: LGBM0.4/SVR0.3/RF0.3、dry: exp0.9/quad0.1)を固定。

## LOSO 結果

| モデル | overall-RMSE | macro-RMSE | ベイスギ(最高MC) |
|---|---|---|---|
| cf 単独 | 22.41 | 18.50 | 82.4 |
| dry 単独 | 27.51 | 25.86 | 61.2 |
| **動的アンサンブル** | **20.08** | **17.37** | **68.9** |
| 参考: 固定 0.5/0.5 | 22.10 | 20.33 | — |

樹種別RMSE(平均MC昇順、抜粋):

| 樹種 | 平均MC | cf | dry | 動的ens |
|---|---|---|---|---|
| ウエンジ | 15 | 23.9 | 5.2 | 23.9 |
| スプルース | 37 | 6.9 | 28.2 | 7.0 |
| ベイマツ | 42 | 5.0 | 23.1 | 5.2 |
| ウォールナット | 57 | 12.7 | 39.9 | 12.7 |
| クリ | 73 | 8.6 | 24.0 | 8.6 |
| イチョウ | 78 | 15.4 | 34.3 | 14.0 |
| **ベイスギ** | **145** | **82.4** | 61.2 | **68.9** |

## 所見

- **動的アンサンブルは cf 単独より macro(18.50→17.37)・overall(22.41→20.08)とも改善**し、
  改善はほぼ全て**高MCのベイスギ(82.4→68.9)に集中**。中低MC樹種は cf 値をそのまま維持。
  =「高MCほど dry を重視」という設計が意図通り機能。
- ゲートは高MC側で**急峻に切り替わる**解が macro 最適(ベイスギのみ dry に寄せるのが macro 的に得なため)。
  滑らかな運用がよければ center/scale を手動で緩める。固定50:50は中域を崩し劣化(macro 20.33)。
- **リーク注意**: cf の特徴選択・dry のバンド選定・本結合の center/scale はいずれも全fold集計で固定＝楽観値。
  厳密な外挿見積りには fold 内で全選択を行うネストLOSOが必要
  ([loso-validation のネスト化]、[collinear_filtered_ensemble.md](collinear_filtered_ensemble.md) 等と同じ構図)。
- 共通入力 `train_clean.csv`(ベイスギ外れ値52行除去後)。dry をドキュメント通りの良好な形(ベイスギ61)で使うため。
  ※全 `train.csv` で実施すると外れ値が exp 当てはめを壊し dry のベイスギが 226 まで悪化する。

## 再現

`notebooks/dynamic_weighted_cf_dry_ensemble.ipynb`:
- cf特徴セル(`CORR_TH=0.90`, `MC_TH=0.30`)→cf OOF、dry exp/quad OOF、動的重みグリッド探索、
  「動的重み曲線＋樹種別RMSE」図、「樹種別 真値 vs LOSO予測(横軸=樹種内サンプル番号0始まり)」図。
- データ Shift-JIS、メタ列 `sample number, species number, 樹種, 含水率`、以降が波数(cm⁻¹)降順の吸光度。
