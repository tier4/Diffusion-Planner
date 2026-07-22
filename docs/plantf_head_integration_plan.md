# planTF ヘッド移植プラン: DP エンコーダ + planTF デコーダの新モデル開発

> **実装状況** (feat/plantf-decoder-head, dev ベース):
> Phase 1・2 は実装済み。`decoder_type="plantf"` で
> `diffusion_planner/model/module/plantf_decoder.py` の `PlanTFDecoder` /
> `compute_plantf_training_loss` が使われる。テストは
> `diffusion_planner/tests/test_plantf_decoder.py`。
> ONNX export(`utils/onnx_export.py` / `ros_scripts/torch2onnx.py`)も実装済み。
> Phase 3 のミニデータセット学習・A/B 比較は未着手。
> Autoware ノードとの ONNX 互換性は §8 を参照
> (full.onnx × `single_step` はノード無変更で互換、multi_step 経路は非対応)。

## 1. 目的

Diffusion Planner (DP) の入出力仕様・エンコーダをそのまま共通利用し、デコーダ(ヘッド)部分だけを
[planTF](https://github.com/jchengai/planTF) (Cheng et al., "Rethinking Imitation-based Planner for Autonomous Driving", ICRA 2024)
のマルチモーダル回帰ヘッドに置き換えた新モデルを開発する。

移植方針は「planTF を DP の仕様に寄せる」。すなわち:

- 入力 dict(`ego_agent_past`, `neighbor_agents_past`, `lanes`, ... )は DP のまま変更しない
- エンコーダ(`Encoder`, MLP-Mixer + FusionEncoder)は DP のまま変更しない
- 出力インターフェース(`decoder_outputs["prediction"]: [B, P, T, 4]` + `turn_indicator_logit`)も DP に合わせる
- デコーダのみ、DiT + 拡散サンプリング → planTF 式の one-shot マルチモーダル回帰に置換する

期待される効果:

- **推論の高速化**: 拡散の反復デノイズ(DPM-Solver 10 ステップ)が 1 回の forward になる
- **学習の単純化・比較実験**: 拡散ヘッドと回帰ヘッドを同一エンコーダ・同一データで A/B 比較できる
- **マルチモーダル出力**: モード確率つきの複数軌道候補が得られる(現状の DP は単一モード)

## 2. 両モデルの構造比較

| 項目 | DP (現状) | planTF | 新モデル |
|---|---|---|---|
| 入力 | dict (ego/neighbors/lanes/route/...) | nuPlan feature dict | **DP と同一** |
| エンコーダ | MLP-Mixer 系サブエンコーダ + SelfAttention 融合 6 層, `hidden_dim=256` | NAT 1D-CNN + TransformerEncoder 4 層, `dim=128` | **DP と同一** |
| エンコーダ出力 | `[B, token_num, 256]`(ego=token 0, 以降 neighbors, static, lanes, ...) | `[B, A+M, 128]`(ego=token 0) | **DP と同一** |
| ego ヘッド | DiT(adaLN-Zero + cross-attn)+ DPM-Solver / flow matching | `TrajectoryDecoder`: ego トークン → `num_modes` 分岐 MLP → 軌道 + モード確率 | planTF 式 |
| neighbor ヘッド | DiT で ego と同時デノイズ(joint) | `agent_predictor`: エージェントトークン → MLP → `(T, 2)` 単一モード | planTF 式(出力は 4ch に拡張) |
| 出力 | `prediction [B, 1+P_n, 80, 4]` (x,y,cos,sin) 単一モード | ego `(6, 80, 4)` + 確率 `(6,)`、neighbor `(T, 2)` | 学習時: マルチモード / 推論時: best mode を選び **DP と同形状** |
| 損失 | 拡散再構成損失 + road border / collision ペナルティ + turn indicator CE | WTA (best-mode) smooth L1 + モード CE + neighbor smooth L1 | planTF 式 + DP のペナルティ類を流用 |
| 推論 | DPM-Solver 10 step / Euler 10 step | 1 forward, argmax mode | 1 forward, argmax mode |

### planTF ヘッドの要点(移植対象)

`TrajectoryDecoder` (planTF `src/models/planTF/modules/trajectory_decoder.py`):

```
x_ego [B, dim]
  → multimodal_proj: Linear(dim → num_modes*dim) → [B, K, dim]
  → loc head:  MLP(dim → 2*dim → T*4)   → [B, K, T, 4]   # x, y, cosθ, sinθ
  → pi head:   MLP(dim → 2*dim → 1)     → [B, K]          # モードロジット
```

`agent_predictor` (planTF `planning_model.py`):

```
x_agents [B, A-1, dim] → MLP(dim → 2*dim → T*2) → [B, A-1, T, 2]
```

損失 (planTF `lightning_trainer.py: _compute_objectives`):

```
best_mode      = argmin_k Σ_t ‖traj_k(t) - gt(t)‖   (xy の ADE で決定)
ego_reg_loss   = smooth_l1(traj[best_mode], gt)      # gt = [x, y, cosθ, sinθ]
ego_cls_loss   = cross_entropy(pi, best_mode)
agent_reg_loss = smooth_l1(pred[valid], gt_xy[valid])
loss = ego_reg_loss + ego_cls_loss + agent_reg_loss
```

## 3. 設計方針

### 3.1 モデルの組み込み方: `decoder_type` スイッチ

DP には model registry がなく、全呼び出し箇所(35 箇所)が `Diffusion_Planner(config)` を直接生成している。
また `utils/config.py` の `Config` は `args.json` の全キーを `setattr` するため、
新しい config フィールドは推論側に自動で伝搬する。

そこで新規クラスを別系統として作るのではなく、`Diffusion_Planner.__init__`
(`diffusion_planner/model/diffusion_planner.py:7`)に分岐を追加する:

```python
class Diffusion_Planner(nn.Module):
    def __init__(self, config):
        self.encoder = Encoder(config)
        if getattr(config, "decoder_type", "diffusion") == "plantf":
            self.decoder = PlanTFDecoder(config)
        else:
            self.decoder = Decoder(config)
```

これにより train.py / valid_predictor.py / simulate.py / ROS ノード / ONNX export の
既存配線を最大限に再利用できる。

### 3.2 デコーダの入出力契約(DP 準拠)

`Decoder.forward(encoder_outputs, inputs)` と同一シグネチャにする。

- 入力: `encoder_outputs [B, token_num, 256]` + `inputs` dict
- 学習時の出力 dict:
  - `trajectory: [B, K, future_len, 4]` — ego のマルチモード軌道(正規化空間)
  - `probability: [B, K]` — モードロジット
  - `neighbor_prediction: [B, P_n, future_len, 4]` — neighbor 単一モード予測
  - `turn_indicator_logit: [B, 5]`
- 推論時の出力 dict(既存の消費側と互換):
  - `prediction: [B, 1+P_n, future_len, 4]` — best mode の ego + neighbor、**denormalize 済み**
  - `turn_indicator_logit: [B, 5]`
  - 追加で `trajectory` / `probability` も返す(可視化・評価用)

拡散専用入力(`sampled_trajectories`, `diffusion_time`, `delay`)は **使わない**(無視する)。
これにより呼び出し側は入力 dict を変えずに済む。

### 3.3 トークンの取り出し

DP エンコーダの token 並びは ego(1) → neighbors(P_n) → static → lanes → route → polygons →
line_strings → goal → ego_shape → turn_indicator(`encoder.py:55-66, 253-267`)。

- ego ヘッド入力: `encoder_outputs[:, 0]`
- neighbor ヘッド入力: `encoder_outputs[:, 1 : 1 + predicted_neighbor_num]`

planTF が ego トークンだけから `num_modes` 本を分岐生成するのと同型で、追加の cross-attention は不要
(まず素の planTF 構造で立ち上げ、性能次第で Phase 4 の拡張を検討)。

### 3.4 座標系・正規化

- DP は GT/予測軌道に `StateNormalizer`(`utils/normalizer.py:8`)を適用して学習し、推論で inverse する。
  新ヘッドも同じ流儀に従う: **正規化空間で回帰・損失計算し、推論時に `state_normalizer.inverse` で戻す**。
  これで既存のスケール感・チェックポイント運用と揃う。
- 出力チャネルは DP の `POSE_DIM=4` (x, y, cosθ, sinθ) とし、planTF の out_channels=4 と一致。
  推論での heading 復元は planTF 同様 `atan2(sin, cos)` …だが DP の下流は (x,y,cos,sin) のまま消費するので変換不要。

### 3.5 neighbor ヘッドの出力次元

planTF の agent_predictor は `(T, 2)` (xy のみ) だが、DP の `prediction` テンソルおよび
`neighbor_prediction_loss` は 4ch (x,y,cos,sin) を前提とする。
→ **出力を `(T, 4)` に拡張**し、損失は xy + heading(cos/sin)に対して計算する
(DP の既存 `loss_func` の neighbor 部分をそのまま流用できる形にする)。

### 3.6 planTF 固有機能の扱い

| planTF の機能 | 扱い |
|---|---|
| `TrajectoryDecoder`(マルチモード ego ヘッド) | **移植する**(本プランの中心) |
| `agent_predictor`(neighbor MLP ヘッド) | **移植する**(4ch に拡張) |
| WTA + CE 損失 | **移植する** |
| `StateAttentionEncoder` + state dropout (SDE)(ego 状態過依存の抑制) | エンコーダ変更になるため **初期スコープ外**。Phase 4 のオプション(DP は既に `NeighborDropoutAugmentation` 等の入力レベル augmentation を持ち、それらはそのまま併用可能) |
| NAT 履歴エンコーダ / MapEncoder | 移植しない(DP エンコーダを使う) |
| minADE/minFDE/MR メトリクス | 検証用に**簡易移植する**(torchmetrics 依存にはせず、validation 内で計算) |

### 3.7 DP 側機能との整合

- **turn indicator ヘッド**: DP の `turn_indicator_predictor`(`decoder.py:290`)を新デコーダにも搭載。
  入力は「best mode の ego 軌道サブサンプル + pooled encoding」で既存実装を流用。
- **road border / collision ペナルティ**(`loss.py:288, 394`): ego 軌道に対する幾何ペナルティであり
  拡散に依存しないため、**best mode 軌道に対して適用可能**。係数は config で 0 にもできる形で流用。
- **delay / prefix constraint**: 拡散のデノイズループ中にプレフィックスを固定する仕組みで、
  回帰ヘッドには構造的に適用できない。初期実装では無視する(推論側の `delay` 入力は受け取るが未使用)。
  必要になれば「先頭 `delay` ステップを現在状態で上書き」する後処理で近似(Phase 4)。
- **flow matching / guidance**: 拡散専用機能のため `decoder_type="plantf"` では非対応。
  config バリデーションで組み合わせ違反を明示エラーにする。

## 4. 実装ステップ

### Phase 1: モデル本体

新規ファイル `diffusion_planner/diffusion_planner/model/module/plantf_decoder.py`:

- `PlanTFTrajectoryHead`: planTF `TrajectoryDecoder` の移植
  (`embed_dim=hidden_dim(256)`, `num_modes=K(デフォルト6)`, `future_steps=future_len(80)`, `out_channels=4`)
- `PlanTFAgentPredictor`: `build_mlp(dim, [2*dim, T*4], norm="ln")` 相当
- `PlanTFDecoder(nn.Module)`: 上記 2 ヘッド + turn indicator ヘッドを束ね、
  `forward(encoder_outputs, inputs)` で §3.2 の契約を実装。
  `self.training` で学習/推論出力を切り替え(DP `Decoder.forward` と同じ流儀)
- 重み初期化は planTF の `_init_weights`(xavier_uniform)を移植

変更ファイル:

- `model/diffusion_planner.py`: `decoder_type` 分岐(§3.1)。`sde` プロパティ等、拡散前提の属性アクセスをガード
- `train_config.py`: 追加フィールド
  - `decoder_type: str = "diffusion"`(`"diffusion" | "plantf"`)
  - `num_modes: int = 6`
  - `alpha_mode_cls_loss: float = 1.0`(モード CE の係数)
  - 既存の `alpha_planning_loss` / `alpha_neighbor_loss` / `coeff_road_border` / `coeff_neighbor_collision` は流用

### Phase 2: 損失・学習ループ

- `plantf_decoder.py`(または `model/module/plantf_loss.py`)に `compute_plantf_training_loss` を実装:
  1. GT 構築・`state_normalizer` 適用は DP `compute_training_loss`(`decoder.py:65`)の前段処理を流用
     (拡散 time サンプリング・ノイズ付与・prefix mask はスキップ)
  2. WTA best mode 選択(xy ADE、valid mask 考慮)→ ego smooth L1(4ch)
  3. モード CE
  4. neighbor smooth L1(valid mask 考慮。DP の per-agent valid 判定を流用)
  5. best mode 軌道に対する road border / collision ペナルティ(denormalize してから既存関数へ)
  6. turn indicator CE(既存流用)
- `train_epoch.py`: `decoder_type` で `compute_training_loss` / `compute_plantf_training_loss` を分岐し、
  損失合成(`train_epoch.py:86-92`)に `alpha_mode_cls_loss * cls_loss` を追加。
  augmentation・`observation_normalizer`・EMA・grad clip はそのまま共通

### Phase 3: 推論・評価・デプロイ

- `PlanTFDecoder` の推論分岐: `probability.argmax` → best mode 選択 →
  neighbor 予測と連結して `prediction [B, 1+P_n, T, 4]` を構成 → `state_normalizer.inverse`
- `valid_predictor.py` / `simulate.py` / ROS ノード: 出力契約が同じため原則無変更で動くことを確認。
  拡散前提の処理(サンプリングループ、`sampled_trajectories` 生成)は入力 dict に残っていても無害
- validation メトリクスとして minADE_K / minFDE_K / MR を追加(planTF 準拠、`decoder_type="plantf"` 時のみ)
- ONNX: 反復デノイズが無いため `FullONNXWrapper`(`onnx_export.py:222`)相当の単一グラフで完結。
  `onnx_export.py` に `decoder_type` 分岐を追加(Encoder ONNX は共通、Decoder は 1-shot 版ラッパー)

### Phase 4(オプション・性能次第)

- planTF の **state dropout encoder (SDE)** 相当を DP の `EgoEncoder` にオプション追加
  (ego 現在状態チャネルへの attention + 学習時ランダムマスク)。planTF の主要な貢献の一つで、
  closed-loop での ego 状態過依存(ego progress への過適合)を抑える
- ego ヘッドを「モードクエリが encoder トークンへ cross-attention する」構造に拡張
  (`exploration_policy/module/heads.py` の `GuidanceHead` が実装テンプレートになる)
- delay/prefix の後処理近似
- 蒸留・アンサンブル等、拡散ヘッドとのハイブリッド

## 5. 検証計画

1. **単体テスト**(uv、`docs/unit_testing_with_uv.md` の流儀):
   - `PlanTFDecoder` の forward 形状テスト(学習/推論両モード、ダミー入力は `onnx_export.py` の生成関数を流用)
   - 損失関数のテスト: GT をモード k に一致させたとき best_mode==k、損失 ≈ 0 になること、
     valid mask ゼロの neighbor が損失に寄与しないこと
2. **ミニデータセット学習**(`docs/training_quickstart_mini_dataset.md`):
   loss 収束・minADE/minFDE の推移を確認、過学習で GT をほぼ再現できること(sanity check)
3. **本学習 + A/B 比較**: 同一データ・同一エンコーダ設定で `decoder_type=diffusion` vs `plantf` を学習し、
   open-loop 指標(ADE/FDE/MR、turn indicator acc)と推論レイテンシを比較
4. **closed-loop / シミュレーション**: `simulate.py` 系での挙動確認、ROS ノードでの実機系確認

## 6. リスク・論点

| リスク | 対応 |
|---|---|
| DP エンコーダは拡散ヘッド前提で学習された設計(トークン粒度・正規化)であり、回帰ヘッドで性能が出ない可能性 | Phase 3 の A/B で早期に判断。必要なら Phase 4 の cross-attention 化・SDE 追加 |
| 単一モード WTA 学習はモード崩壊しやすい | planTF 準拠の CE + `num_modes` チューニング。将来的に winner-take-all の soft 化も検討 |
| neighbor を 4ch に拡張した際の heading 学習が不安定になる可能性 | heading 項の損失重みを config で調整可能にする(0 で planTF と等価) |
| `compute_training_loss` が decoder モジュールと密結合 | planTF 用損失は別関数として実装し、train_epoch で分岐(既存コードへの侵襲を最小化) |
| ONNX/ROS 側に拡散前提の暗黙依存が残っている可能性 | Phase 3 で `decoder_type=plantf` の e2e 通しテストを行い、洗い出す |

## 7. 参考

- planTF: https://github.com/jchengai/planTF — 特に
  `src/models/planTF/planning_model.py`, `modules/trajectory_decoder.py`,
  `modules/agent_encoder.py`(SDE), `lightning_trainer.py`(損失)
- DP 側の接続点(dev ブランチのレイアウト):
  - `diffusion_planner/model/diffusion_planner.py`(Encoder/Decoder 合成、`build_decoder` で分岐)
  - `diffusion_planner/model/module/encoder.py`(トークン並び: ego=0, neighbors=1〜)
  - `diffusion_planner/model/module/decoder.py`(既存の拡散デコーダと `compute_training_loss`)
  - `diffusion_planner/train_epoch.py`(損失合成、decoder_type で損失関数を分岐)

## 8. Autoware ノードとの ONNX 互換性調査(2026-07-17)

`autoware_universe/planning/autoware_diffusion_planner`(ROS ノード)のコードを読み、
`decoder_type=plantf` で export した ONNX がそのままデプロイできるかを調査した。
ノード側の接続点は以下(パスは autoware_universe リポジトリ内):

- 入出力次元の定義: `include/autoware/diffusion_planner/dimensions.hpp`
- 推論バックエンド選択: `src/diffusion_planner_core.cpp:126-154`
  (`model.type` = `single_step` | `multi_step` × `model.backend` = `tensorrt` | `ort_*`)
- TensorRT single-step: `src/inference/single_step_inference.cpp`
- ONNX Runtime single-step / multi-step: `src/inference/onnxruntime_inference.cpp`
- TensorRT multi-step: `src/inference/multi_step_inference.cpp`

### 8.1 結論サマリ

| ONNX / ノード経路 | 互換性 | 備考 |
|---|---|---|
| encoder.onnx | **互換** | 拡散版と入出力契約が完全に同一(入力 14 本 → `encoding [B, 564, 256]`) |
| full.onnx × `single_step`(TRT / ORT) | **互換(対策実装済み)** | 未使用入力 3 本が export 時にグラフから消える問題があり、`PlanTFFullONNXWrapper` で解消(§8.2) |
| decoder.onnx × `multi_step`(TRT / ORT) | **非互換(設計上)** | multi-step 経路は DPM-Solver ループ前提でグラフ契約が根本的に異なる(§8.3) |

### 8.2 full.onnx × single_step: 未使用入力の削除問題

ノードの single-step 経路は **拡散版 full.onnx の全 17 入力**を feed する:

- ORT 版(`onnxruntime_inference.cpp:92-110`)は `sampled_trajectories`〜`delay` の 15 float +
  speed limit mask 2 bool を `session_.Run` に渡す。ORT は**グラフに存在しない入力名の feed を
  `INVALID_ARGUMENT` で拒否**する(手元の ORT で実験確認済み)。
- TRT 版(`single_step_inference.cpp:121-137, 152-203`)は 17 入力すべてに
  `setInputShape` / `setTensorAddress` するため、グラフに無いテンソル名でエンジン設定が失敗する。

一方 planTF ヘッドは `sampled_trajectories` / `ego_current_state` / `delay` を一切参照しない
(§3.2。`ego_current_state` は拡散デコーダの prefix constraint 専用で、エンコーダも使わない)。
レガシー exporter(`dynamo=False`)は**出力に寄与しない入力をグラフから削除する**ため、
planTF の full.onnx は入力 14 本になる。小型 config の実モデル export で確認した:

```
kept inputs : [ego_agent_past, neighbor_agents_past, static_objects, lanes, ..., turn_indicators]  # 14 本
dropped     : [sampled_trajectories, ego_current_state, delay]
outputs     : prediction [batch, 321, 80, 4], turn_indicator_logit [batch, 5]
```

出力側はノードの期待(`OUTPUT_SHAPE {1, 321, 80, 4}` / `TURN_INDICATOR_LOGIT_SHAPE {1, 5}`、
出力名 `prediction` / `turn_indicator_logit`、denormalize 済み、batch 軸 dynamic)と一致しており、
**入力契約だけが壊れている**。なお同じ理由で、修正前は `torch2onnx.py` の
`validate_full_model`(全 17 入力を feed)も planTF チェックポイントで失敗する。

**対応(実装済み)**: export 側で残す(ノード無変更・変更最小)。
`utils/onnx_export.py` の `PlanTFFullONNXWrapper` が未使用 3 入力をゼロ係数の残差でダミー消費
(`prediction + 0.0 * (sampled_trajectories[:, 0, 0, 0] + ego_current_state[:, 0] + delay[:, 0])`)
してグラフに 17 入力を保持する。全和ではなくスカラースライスを使うのは、fp16 等の
低精度ランタイムで dead branch が overflow → NaN 化して `prediction` を汚染しないため。
検証済み:

- 小型 config の実モデル export で 17/17 入力が保持される(onnxsim 通過後も)
- ノードと同じ「全 17 入力を名前で feed」する ORT 実行が成功し、出力は torch と一致
  (max diff ~1e-6)。keep-alive 残差は fp32 で厳密に 0 なので出力はビット単位で不変
- 回帰テスト: `tests/test_plantf_decoder.py::test_full_onnx_wrapper_keeps_diffusion_only_inputs`

これにより `torch2onnx.py` の `validate_full_model`(全 17 入力を feed)も planTF
チェックポイントで通るようになる。

別案としてノード側に「planTF 用入力セット(14 本)」の single-step 経路を追加する手も
あるが、ノード改修が必要なため採らなかった。

### 8.3 decoder.onnx × multi_step: 設計上の非互換

ノードの multi-step 経路は拡散専用の構造を持つ:

- コンストラクタが encoder / decoder / **turn_indicator の 3 モデルを必須**とする
  (`onnxruntime_inference.cpp:371-384`)が、planTF export は turn_indicator.onnx を生成しない
- decoder へ `{encoding, sampled_trajectories, diffusion_time, neighbor_agents_past}` を feed し
  出力 `model_output` を取り出して **DPM-Solver ループを C++ 側で回す**
  (`onnxruntime_inference.cpp:432-460`)。planTF の decoder.onnx は
  `encoding → (prediction, probability, turn_indicator_logit)` の 1 コール契約で全く合わない
- ループ内の prefix constraint(`apply_prefix_constraint`)や guidance
  (`src/inference/guidance/`、stop / start / centerline)もデノイズループ前提

planTF の分割グラフを使うには、ノード側に「encoder → decoder 1 コール」の
one-shot split 推論クラス(`is_denormalized = true` で返す)を新設する必要がある。
当面は §8.2 の修正を入れた full.onnx × `single_step` でのデプロイが最短経路。

### 8.4 互換でも挙動が変わる点(planTF では機能しないノード機能)

- **delay 補正**: ノードは `delay_step` を計算して `delay` 入力に渡し
  (`diffusion_planner_core.cpp:473-476`)、拡散デコーダが先頭ステップを prefix constraint で
  固定する。planTF は `delay` を無視するため、入力を残しても**この補正は no-op** になる
  (§3.7 の delay 方針どおり。必要なら後処理近似を Phase 4 で検討)
- **guidance**: multi-step 専用機能のため使用不可
- **デノイズ過程の可視化**(`denoising_predictions`): 反復が無いため出力されない
- **モード確率**: full.onnx の出力は `prediction` / `turn_indicator_logit` の 2 本固定で
  `probability` を含まない(ノードも現状消費しない)。マルチモード候補をノードで使う場合は
  出力契約の拡張が必要

### 8.5 前提条件

- `predicted_neighbor_num=320`(デフォルト、`MAX_NUM_NEIGHBORS`)で学習すること。
  `prediction` の agent 軸が `1 + predicted_neighbor_num` になるため、これ以外だと
  ノードの `OUTPUT_SHAPE {1, 321, 80, 4}` と合わない(拡散版と同じ制約)
- `num_modes` は full.onnx の出力形状に現れないため自由(グラフ内部で argmax 済み)
