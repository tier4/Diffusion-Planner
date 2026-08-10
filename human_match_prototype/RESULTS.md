# Human-match prototype: results

Does the planner's sampled trajectory distribution agree with what the human
actually did? We scored 1454 curated risk frames (`or_scene`: stopped-vehicle,
avoidance, and "can't-make-the-turn" scenarios) and 500 normal-driving frames,
drawing N=64 ONNX samples per frame at a fixed seed, and compared three
candidate mismatch signals:

- **`min_ade_4s`** — coverage: min average displacement error (over 4 s) between
  any of the 64 samples and the human path. Higher = no sample matched the human.
- **`maha_pct`** — typicality: empirical percentile of the scene's Mahalanobis
  distance in encoder-latent space (shared fit). Higher = less typical.
- **`latent_knn_mean`** — OOD: mean distance to k nearest neighbors in latent
  space. Higher = more out-of-distribution scene.

All numbers below are from `data/human_match/scores_{or,normal}.csv` (0 frames
skipped in either run) and `data/human_match/analysis/summary.md`.

## 1. Does any signal separate risk from normal driving?

Yes — coverage separates cleanly, OOD separates moderately, typicality does not.

| signal | risk median [p90] | normal median [p90] | verdict |
|---|---|---|---|
| `min_ade_4s` | 0.484 [2.023] | 0.106 [0.366] | strong (~4.6x median, ~5.5x p90) |
| `latent_knn_mean` | 0.772 [0.858] | 0.537 [0.691] | moderate, consistent shift |
| `maha_pct` | 100.0 [100.0] | 100.0 [100.0] | **saturated — no signal** |

The **mismatch rate** (`frac_close_4s == 0`, i.e. not one of the 64 samples came
within the closeness threshold of the human) is the sharpest separator:

- Risk: **341 / 1454 = 23.5%** full mismatches. Normal: **5 / 500 = 1.0%**.
- The gap widens with horizon: 2 s → 326 vs 1; 4 s → 341 vs 5; 8 s → 494 vs 13.

By risk category the mismatch rate is:

| category | mismatch_4s rate |
|---|---|
| 回避_車両 (avoidance) | 29.4% |
| 曲がり切れない (can't complete turn) | 25.5% |
| 停止_車両 (stopped vehicle) | 15.6% |

**`maha_pct` is dead on arrival**: 1452/1454 risk frames and 500/500 normal
frames sit at exactly 100.0 (std 0.06, min 98.44). This is a *structural*
property of per-scene shared-fit Mahalanobis on tight clouds, not a fixable fit
problem. The LedoitWolf covariance in `typicality.py` is fit per scene on that
scene's 64 samples (a shared fit, not leave-one-out), so with the sample clouds
this tight (median `spread_4s` = 0.309 m) the distance normalizes by a tiny
covariance and even a sub-meter human offset exceeds every in-sample distance —
the percentile saturates by design. The human genuinely sits outside the tight
cloud (the same mismatch `min_ade` measures); `maha_pct` = 100 means the scene is
atypical, not typical. There is no cross-scene/global/per-population fit here that
could go stale — the only prospective fixes are leave-one-out scoring or a
non-saturating distance.

## 2. Do the three signals agree?

Partially. Spearman correlation on the risk set:

|  | min_ade_4s | maha_pct | latent_knn_mean |
|---|---|---|---|
| min_ade_4s | 1.00 | -0.05 | **0.62** |
| maha_pct | -0.05 | 1.00 | -0.06 |
| latent_knn_mean | 0.62 | -0.06 | 1.00 |

- `min_ade_4s` and `latent_knn_mean` are moderately-to-strongly monotonically
  related (0.62): OOD-looking scenes tend to be harder to human-match. But their
  **top-20 sets overlap only 1/20** — they agree on the trend, not on which
  handful of scenes are the single worst. They are complementary, not redundant.
- `maha_pct` correlates ~0 with both and overlaps 0/20 — but that is a
  consequence of its by-design saturation on tight clouds, not independent
  information.

## 3. Are the top mismatches lateral (path choice) or longitudinal (timing)?

**Overwhelmingly longitudinal (speed/timing), not lateral (path choice).**

- Top-20 by `min_ade_4s`: mean |`best_lon_err_4s`| = **3.76 m** vs
  mean |`best_lat_err_4s`| = **0.22 m**; lateral error exceeds longitudinal in
  **0 / 20** of them. Example rows: ade 4.37 (lon 4.37 / lat 0.05),
  ade 3.63 (lon 3.63 / lat 0.31).
- Across the whole risk set the same asymmetry holds: mean |lon| 0.598 vs
  |lat| 0.288; mean |speed err| 0.418 (m/s) vs 0.107 in normal.

Eyeballed overlays confirm the story:

- **曲がり切れない #044**: human drives ~40 m forward in 4 s; the 64-sample cloud
  stays collapsed near the ego. The planner is far more conservative and expects
  a near-stop.
- **回避 #473 / #484**: the sample cloud follows the *same lateral curve* as the
  human but only reaches x≈27–28 m while the human reaches x≈56–57 m — the human
  is ~2x faster over the same 4 s. Path shape agrees; speed does not.
- **normal (moving) #05599**: the sample cloud sits directly on top of the human
  path all the way to x≈40 m — the "match" baseline.

So the planner rarely disagrees with humans about *which way to go*; it disagrees
about *how fast / whether to commit* — braking or hesitating where the human
accelerates through avoidance and tight-turn situations.

## 4. Recommendation: which signal(s) to keep?

**Keep `min_ade_4s` (coverage) as the primary signal; keep `latent_knn_mean` as
a secondary/complementary OOD flag; drop `maha_pct`.**

- **`min_ade_4s` / mismatch rate** is the clear winner: strong risk-vs-normal
  separation, directly interpretable, and via the lat/lon decomposition it tells
  you *why* (here: longitudinal). Report it multi-horizon (2/4/8 s) since the gap
  grows with horizon.
- **`latent_knn_mean`** is worth keeping as a cheap, model-internal OOD flag: it
  needs no human future, separates moderately, and its 1/20 top-set overlap with
  coverage means it surfaces a partly different set of scenes worth auditing.
- **`maha_pct`** should not be used — it saturates at 100 for both populations
  by design (per-scene shared-fit Mahalanobis on tight clouds; see §1). This is
  not a stale fit that can be refit away: the raw distance runs the wrong way
  too (risk median `maha_dist` ≈ 2110 vs normal ≈ 5210, with extreme tails), so
  the typicality signal can't be rescued by skipping the percentile either. If
  ever revisited, leave-one-out scoring or a non-saturating distance would be the
  place to start.

## Caveats

- **Tight sample clouds are systematic, and they make coverage a demanding test.**
  Median `spread_4s` is 0.309 m (risk) / 0.166 m (normal) — the 64 samples cluster
  tightly, consistent with the ~0.7 m endpoint spread seen on one smoke frame
  being at the high end rather than typical. Because the samples move together, a
  wrong mode means *all* 64 miss at once, so `frac_close`/`min_ade` behaves close
  to a single-sample check. This is a strength for flagging (few false "matches")
  but means these thresholds are not measuring distributional spread — with such
  tight clouds `frac_close` is essentially "did the single planner mode match?".
- **Fixed seed (0).** All scores and overlays are deterministic at one seed; the
  fixed-seed self-check artifact from Task 5 confirms determinism but this does
  not characterize seed-to-seed variance of the sample cloud.
- **`maha_pct` saturation** (above) is a structural consequence of per-scene
  shared-fit Mahalanobis on tight sample clouds, not a stale/mismatched fit — the
  human legitimately lands outside the tight cloud, so the percentile pins at 100
  by design. Only leave-one-out scoring or a non-saturating distance could change
  that.
- **Some normal frames are near-stationary** (ego parked/stopped; sub-0.02 m
  scale). These trivially "match" and dilute the normal-set spread slightly, but
  do not affect the risk-vs-normal conclusion given the size of the gap.
- Only the `or_scene` risk set is category-labeled; the normal set is a single
  pool. Category-level conclusions apply to the risk set only.
