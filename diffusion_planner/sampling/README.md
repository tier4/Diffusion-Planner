# Sampling Package

## 概要

Sampling Package は Diffusion Planner の学習に利用するデータの偏りを避けるため、データが適切な分布となるようにサンプリングを行うためのパッケージです。

Sampling Package は下記のファイルから構成されます。

## ファイル構成

```
sampling/
├── cluster.py            # クラスタリング本体（エルボー法 + KMeans）
├── visualize_cluster.py  # クラスタリング結果の可視化
├── utils/
│   └── elbow.py          # WCSS 計算・エルボー検出・KMeans のユーティリティ
└── README.md
```

### 各ファイルの役割

| ファイル | 役割 |
|---|---|
| `cluster.py` | NPZ ファイル一覧を受け取り、ego の将来軌跡を特徴量として KMeans でクラスタリングする。最適クラスタ数はエルボー法で自動決定する。 |
| `visualize_cluster.py` | `cluster.py` が出力した JSON を読み込み、クラスタごとに ego 将来軌跡を重ね描きしたサブプロット図を出力する。 |
| `utils/elbow.py` | WCSS（クラスタ内二乗和）の計算、エルボー点の検出、KMeans フィットをまとめたユーティリティ。 |

---

## 使い方

### Step 1: クラスタリング（`cluster.py`）

```bash
python cluster.py \
    --data_list /path/to/data_list.json \
    --output    /path/to/result.json \
    [--k_max 20] \
    [--pca_components 50] \
    [--seed 42]
```

| 引数 | 必須 | デフォルト | 説明 |
|---|---|---|---|
| `--data_list` | ✓ | — | NPZ ファイルのパスを列挙した JSON |
| `--output` | ✓ | — | クラスタリング結果を書き出す JSON のパス |
| `--k_max` | | `20` | 評価するクラスタ数の上限 |
| `--pca_components` | | `50` | PCA で削減する次元数 |
| `--seed` | | `42` | 乱数シード |

**入力 JSON 形式（`--data_list`）**

```json
[
    "/path/to/sample_0000.npz",
    "/path/to/sample_0001.npz",
    ...
]
```

### Step 2: 可視化（`visualize_cluster.py`）

```bash
python visualize_cluster.py \
    --cluster_json /path/to/result.json \
    --output       /path/to/figure.png \
    [--max_samples 200] \
    [--seed 42]
```

| 引数 | 必須 | デフォルト | 説明 |
|---|---|---|---|
| `--cluster_json` | ✓ | — | `cluster.py` が出力した JSON |
| `--output` | | — | 保存先のパス（PNG / PDF / SVG）。省略するとインタラクティブ表示 |
| `--max_samples` | | `200` | クラスタごとに描画する軌跡の最大本数 |
| `--seed` | | `42` | サンプリング乱数シード |

---

## 処理パイプライン

```
NPZ ファイル群
     │
     ▼
ego_agent_future (80, 3) をフラット化 → (240,)
     │
     ▼
Z スコア正規化
     │
     ▼
PCA（240 次元 → pca_components 次元）
     │
     ▼
エルボー法で最適 k を決定（k = 1 .. k_max）
     │
     ▼
KMeans（k = optimal_k）
     │
     ▼
result.json（クラスタ ID ごとに NPZ パスを分類）
```

---

## 出力ファイル

### `cluster.py` の出力 JSON

クラスタ ID をキー、そのクラスタに属する NPZ ファイルのパス一覧を値とした辞書形式です。  
キーは `cluster_id0`、`cluster_id1`、… の順にソートされています。

```json
{
    "cluster_id0": [
        "/path/to/sample_0000.npz",
        "/path/to/sample_0003.npz"
    ],
    "cluster_id1": [
        "/path/to/sample_0001.npz"
    ],
    "cluster_id2": [
        "/path/to/sample_0002.npz",
        "/path/to/sample_0004.npz"
    ]
}
```

- 全入力ファイルがいずれか 1 つのクラスタに過不足なく割り当てられます。
- 最適クラスタ数はエルボー法により `[1, k_max]` の範囲で自動決定されます。

### `visualize_cluster.py` の出力図

クラスタ数に応じてサブプロットを格子状に配置した PNG（または PDF / SVG）を出力します。  
各サブプロットには、クラスタに属するサンプルの `(x, y)` 軌跡を重ね描きします。

---

## トラブルシューティング

### `OpenBLAS: Program is Terminated. Because you tried to allocate too many memory regions.`

エルボー法のループ中に OpenBLAS がメモリマップ領域の上限を超えることで発生します。  
実行前に下記の環境変数を設定してください。

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python cluster.py ...
```
