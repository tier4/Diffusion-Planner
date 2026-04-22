# Map Comparison Tool
## マップ変換評価ツール チュートリアル

Diffusion Planner（DP）用マップの編集・検証を行う方向けのガイドです。このツールの目的は、マップ上の「ホットスポット」（近似誤差が大きい箇所）を見つけ、Lanelet / LineString を適切に分割・トリムすることで、DP の近似計算精度を向上させることです。

> DP の内部仕様を知らなくても、ヒートマップの色で修正すべき箇所を特定できます。

---

## 1. 簡易フロー

![image](./docs/flow_arch.png)

### データフロー概要

| 段階 | 説明 |
|------|------|
| **入力** | Lanelet2 形式のマップ（`.osm`）と投影情報（`map_projector_info.yaml`） |
| **map_exporter** | マップを読み込み、Diffusion Planner 用の内部形式に変換。同時に「参照用」の生データも出力 |
| **compare_map** | 内部形式（変換後）と参照（生データ）を比較し、幾何学的な誤差を算出 |
| **出力** | メトリクス、可視化プロット、インタラクティブ HTML ダッシュボード |

---

## 2. 前提条件

### 必要なファイル構成

マップディレクトリには次の 2 ファイルが必要です：

```
マップディレクトリ/
├── lanelet2_map.osm          # Lanelet2 形式のベクターマップ
└── map_projector_info.yaml   # 投影方式の設定（MGRS または TransverseMercator）
```

### `map_projector_info.yaml` の例

**MGRS 形式の場合：**
```yaml
projector_type: MGRS
vertical_datum: WGS84
mgrs_grid: 54SUE
```

**TransverseMercator 形式の場合：**
```yaml
projector_type: TransverseMercator
vertical_datum: WGS84
map_origin:
latitude: 35.674438
longitude: 139.747333
altitude: 0
```

### Python 依存関係

```bash
pip install numpy matplotlib jinja2
```

### CPP Tools build

```bash
git clone git@github.com:tier4/Diffusion-Planner.git
cd cpp_tools
bash prepare_repos.sh
bash build.sh
```

---

## 3. 使い方

### 3.1 ワンショット実行（推奨）

マップをエクスポートし、そのまま評価まで行う方法です。

```bash
# ワークスペースのルートから実行  Diffusion-Planner
source cpp_tools/install/setup.bash
python3 cpp_tools/src/autoware_diffusion_planner_tools/scripts/map_eval/compare_map.py \
    export-eval \
    --map_path \
    /path/to/your/map/lanelet2_map.osm \
    --out_dir ./map_eval_result \
    --lane_threshold 0.5 \
    --line_threshold 0.5 \
    --web
```

- `--map_path` : Lanelet2 マップ（`.osm`）のパス
- `--out_dir` : 評価結果の出力ディレクトリ
- `--web` : 計算完了後、自動的にブラウザでインタラクティブ HTML を起動します（推奨）

![image](./docs/website_base.png)

---

## 4. UI の使い方

### 4.1 Lanelet

`--web` オプションで計算完了後、ブラウザが自動起動します。


| | |
|---|---|
| 左上でLane heatmap・line heatmapを選択できます。<br><br>Laneletをチェックしようとしたら、Lane heatmapを選択 | ![](./docs/select_lane_heatmap.png) |
| 右側ではエラーが高いIDのLaneLetとLineStringが表記されます。 <br>  該当の ID をクリックしたら、自動的に Zoom されます。 | ![](./docs/zoom_top_lane_error.png) |
| 赤い点を選択して、実際のエラーを見えます。 <br>🔴 赤い点：DP近所点<br>🔵 青い点：地図 <br> **修正方法**： VectorMapBuilder で該当する Lanelet を**適切に分割**してください。分割することで DP の近似精度が向上します。  | ![](./docs/lanelet_dot_error.png) |
| 修正済み/交差点に分割し辛い/不要と判断されたLanelet の判断は作業者に任せます。<br> Checkboxをクリックしたら、このSessionで一時隠すことができます。画面 Refresh したら回復。 | ![](./docs/hide_lanelet.png) |
| そうしたら、エラーが更に低いレーンも検証できるように | ![](./docs//lower_error_lanelet.png) |



#### 判断例：

| 画像例 | 対応 |
|------|------|
| ![image](./docs/continuous_curve.png) | 連続なカーブが一つレーンになってしまいました。<br> 修正すべきです。 |
| ![image](./docs/long_intersection_lanelet.png) | このような交差点でのカーブは自動運転に影響があるので、分割は控えて、レポートしていただければ幸いです。 |



---

### 4.2 LineString

| | |
|---|---|
| LineStringをチェックしようとしたら、Line heatmapを選択 <br> 同じように右のTop Error Linesをクリック <br> 🔴 赤い点：DP近所点<br>🔵 青い点：地図 | ![](./docs/select_line_heatmap.png) |


**修正方法**: 該当する LineString を**適切に分割**してください。

**判断の目安**: 主に直線の道路など、誤差が走行に影響しない箇所もあります。その判断は編集者にお任せします。

#### 判断例：

| 画像例 | 対応 |
|------|------|
| ![image](./docs/border_short_cut.png) | BorderはShortcutされた <br> 修正すべきと判断。 |
| ![image](./docs/straight_road_border.png) | 走行に影響のない直線Border、修正しなくても大丈夫 |

---

## 5. ホットスポット修正のワークフロー

1. **評価実行**（`--web` でブラウザ自動起動）
2. **Lane heatmap** でレーンのホットスポットを確認
    - 黄色い箇所をズームし、DP 近似と元地図の違いを確認
    - 該当 Lanelet を適切に分割（交差点右折レーンは無理に分割しなくてよい）
3. **Line heatmap** で Road border のホットスポットを確認
    - 黄色い箇所をズームし、Road border の近似誤差を確認
    - 該当 LineString を適切に分割（直線など影響が小さい箇所は判断に任せる）
4. **マップを編集**
    ```bash
    python3 .../compare_map.py export-eval \
    --map_path /path/to/edited_map/lanelet2_map.osm \
    --out_dir ./map_eval_after_edit \
    --lane_threshold 0.5 --line_threshold 0.5 \
    --web
    ```
5. **再評価して改善を確認**
    - `metrics_summary.json` の `pass_rate` や `Hausdorff` 値が改善しているか確認

---

## 6. よくあるトラブル

### `map_projector_info.yaml` が見つからない

`lanelet2_map.osm` と同じディレクトリに `map_projector_info.yaml` を配置してください。
