# Investigation: does a shared VAE close the gap to persistence?

**Phase 2 / Session 10.** Grounded in real runs of a new script
(`scripts/validate_shared_vae_hypothesis.py`), executed this session against four
real datasets. Every number below was produced by an actual fit-and-score run, not
estimated — the exact commands are listed at the end of this document so anyone can
reproduce them. `gears/evaluation/benchmark.py`, the committed
`results/benchmark/all_results.parquet`, and every model's shipped default
hyperparameters are unchanged by this investigation, per this session's own scope.

## The hypothesis under test

`REFACTOR_STATE.md`'s Phase 1 "Session 3 — VAE competitiveness" write-up proposed,
but never tested, a specific explanation for why persistence-bootstrap keeps beating
GMM and VAE on the existing rolling-origin benchmark: the harness retrains a fresh,
tiny VAE from scratch on each cell's own narrow windowed pool, discarding the VAE's
one real architectural advantage over GMM — a single network that can borrow
statistical strength across many contexts. The question this session tests:
**does fitting ONE shared VAE across all training history up to each rolling
origin, then scoring it against each held-out day, close the gap to persistence —
where the current per-cell-retrain design structurally cannot?**

## Method

New script, `scripts/validate_shared_vae_hypothesis.py` (does not modify
`benchmark.py`'s core logic, same precedent as `validate_recency_bias.py` /
`validate_vae_competitiveness.py`). For each rolling origin: fits ONE
`EVSessionModel(model_type="vae", stratify_by=["day_of_week"])` on all of that
origin's training history (`vae_shared`), reused across every `day_offset` and every
`X`. As a direct control, on the exact same cells, also fits fresh `persistence`
(`PersistenceSessionSampler`) and the existing per-cell `vae_percell` arm
(identical hyperparameters to `vae_shared`, fit only on the windowed pool) — so any
difference is attributable to the shared-vs-per-cell design, not a hyperparameter
change. Same Wasserstein/CRPS/profile-NRMSE metrics and the same "mean of the three
Wasserstein distances, lower wins" convention as `benchmark.py` / notebook 4.

**Datasets** (4, exceeding the plan's "at least 3"): `office` (1,426 sessions,
small/sparse, workplace charging with almost no weekend activity), `sap` (26,430
sessions, long/continuous), `boulder` (21,569 sessions, the "one large" dataset —
picked over `perth` for tractability, 21.6k vs 63.9k rows), and
`sample_df.pkl` filtered to `location_type="home", department="92"` (42,278
sessions — the same official validation subset Phase 1 Session 3 used).

**Grid actually run — deliberately reduced, stated explicitly, not silently:**
`X ∈ {4, 16}` (office also got `X=1`), `horizons ∈ {1, 2}` (office also got `3`),
`n_origins = 2` (office got 5, sample_df got 1), `n_scenarios = 10–15`,
`vae_epochs=40–50, hidden_dim=256, latent_dim=16` (the "bigger network" config
Phase 1 Session 3 already found helped most). This is smaller than the full
`X∈{1,2,3,4,8,16,52}` / daily-origin grid `benchmark.py` sweeps by default — a
single sandbox CPU session cannot run that grid with a real neural-net arm in
either design (see the compute section below), so this reduction follows the same
precedent Sessions 3 and 4 already used and documented for the same reason.
**32 paired cells total** across all four datasets — real, but a small sample;
treat percentages here as directional, not precise.

## Results

### 1. Wasserstein win rate vs. persistence — does sharing close the gap?

| Method | Paired cells | Win rate vs. persistence | Mean score | Persistence's mean score |
|---|---|---|---|---|
| `vae_percell` (existing per-cell design) | 32 | 15.6% | 4.111 | 2.942 |
| `vae_shared` (this session's new arm) | 32 | 18.8% | 3.593 | 2.942 |

**No — not on the primary metric.** `vae_shared` is a small improvement over
`vae_percell` (18.8% vs. 15.6%), but persistence still wins the clear majority of
cells either way. This is the honest, negative-leaning half of the answer: the
shared-fitting hypothesis does not close the gap on Wasserstein distance within
this grid.

### 2. Does sharing help relative to the existing per-cell VAE, directly?

| Comparison | Paired cells | `vae_shared` win rate | Mean scores |
|---|---|---|---|
| `vae_shared` vs. `vae_percell` | 32 | **65.6%** | 3.593 vs. 4.111 |

**Yes, clearly.** Independent of whether it beats persistence, `vae_shared` beats
the existing per-cell `vae` design on nearly two-thirds of cells. This is a real,
consistent, positive result — the architectural fix (shared fitting) measurably
improves the VAE arm itself, even though it isn't (yet) enough to overtake
persistence.

### 3. The metric story — profile-NRMSE tells a different, more favorable story

| Method | Paired cells | Profile win rate vs. persistence | Mean NRMSE | Persistence's mean NRMSE |
|---|---|---|---|---|
| `vae_percell` | 32 | 6.2% | 1.460 | 1.069 |
| `vae_shared` | 32 | **34.4%** | 1.239 | 1.069 |

This is the part the plan explicitly asked not to lose as a footnote, and it's the
most interesting finding here. On the aggregate 24h load-profile reconstruction
metric — arguably the more decision-relevant one for GEARS's actual use case
(capacity/load planning, not exact per-session distributional matching) —
`vae_shared` wins over a third of cells, more than five times `vae_percell`'s rate,
and gets meaningfully closer to persistence's mean than it does on Wasserstein.
Session 3's plausible mechanism (individual-session sampling noise partly cancels
out once profiles are aggregated across a full day) is consistent with what's
observed here, and shared fitting appears to amplify that effect rather than just
add capacity. This was not directly re-verified as *the* mechanism this session —
flagged as a good next-session question, not re-litigated here.

### 4. By X — does more history narrow the gap further?

| X | `vae_percell` win rate | `vae_shared` win rate |
|---|---|---|
| 4 | 8.3% (n=12) | 8.3% (n=12) |
| 16 | 20.0% (n=20) | 25.0% (n=20) |

Consistent with Phase 1's own finding (larger X narrows the gap), and `vae_shared`
narrows slightly faster than `vae_percell` as X grows — though both remain
small-sample here (12–20 cells per row) and this should be read as directional.

### 5. Compute cost — a correction to this script's own starting assumption

This script's docstring, written before running anything, argued the shared design
would be "dramatically cheaper" because it fits once per origin instead of once per
cell. **That assumption was wrong, and it's worth stating plainly rather than
quietly fixing the docstring and moving on:**

| Dataset | Sessions | `vae_shared` mean fit time | `vae_percell` mean fit time |
|---|---|---|---|
| office | 1,426 | 3.19s | 0.31s |
| sap | 26,430 | 38.35s | 0.54s |
| boulder | 21,569 | 80.73s | 0.28s |
| sample_df (home, dept 92) | 42,278 | 51.09s | 0.96s |

Per-fit, `vae_shared` is 10–280x slower than `vae_percell`, because it trains on
the *entire* history at each origin while `vae_percell` trains on a tiny windowed
pool. Fitting fewer times doesn't win back the difference: for boulder's actual run
(2 origins × 2 horizons × 2 X), total `vae_shared` compute was **161s** vs.
`vae_percell`'s **9s** — 18x more total compute for the 65.6%-vs-`vae_percell` /
18.8%-vs-persistence result above. The "fewer fits" framing this script started
with was true by *count*, but false by *total wall time*, because pool size (not
fit count) dominates. This is a real, practically-relevant tradeoff to weigh
against the accuracy gain in section 2, not a reason to dismiss it outright — a
smaller network or fewer epochs for the shared arm specifically (it sees far more
data per epoch than a per-cell fit does) is an obvious, untried lever; see next
steps.

## Interpretation, plainly

The shared-fitting hypothesis is **partially confirmed, not fully**. It measurably
improves the VAE arm on its own terms (section 2) and shows a substantially bigger
improvement on the profile metric (section 3) than on the primary one (section 1) —
but it does not close the gap to persistence within this grid, and it currently
costs more total compute than the design it's meant to replace, not less. Reported
as found: this is a real, mixed result, not a reason to change the model that
GEARS's docs/README currently frame as its GMM/VAE default — that recommendation is
Yvenn's call once he's seen this evidence, not something resolved here.

## A documentation-provenance gap noticed along the way (unrelated to this
investigation's own result, flagging rather than fixing)

The Phase 2 plan document cites `results/benchmark/all_results.parquet` as holding
a "committed 4-arm" sweep (persistence 69.5%, vae 13.0%, gmm 9.7%, gmm_recency
7.8%). Checked this session, directly: the file as currently committed contains
**only `persistence` and `gmm`** (779 paired cells, gmm win rate **9.9%** —
consistent with the plan's 9.7% for that pair specifically), not `vae` or
`gmm_recency`. Either an earlier session's 4-arm run was never saved to this exact
path, or the file has been regenerated since. Not investigated further or changed
this session (out of scope, and the plan's own ground rule is ask-don't-guess) —
flagged here for Yvenn to confirm whether a 4-arm `all_results.parquet` exists
elsewhere or should be regenerated.

## Next steps — concrete, scoped, not implemented (Yvenn's call)

1. **Cheaper shared fits**: try fewer epochs and/or a smaller hidden dim
   specifically for the shared arm (it sees 10-40x more data per epoch than a
   per-cell fit, so it plausibly needs far fewer epochs to converge) — could close
   most of the compute gap in section 4 without touching accuracy. Untried this
   session.
2. **Full-precision run on `sap`**: `sap` is the plan's own named "long/continuous"
   candidate and showed the clearest profile-NRMSE signal (50% win rate at X=16)
   with only 4 cells — a same-design run with the full `X∈{1,2,3,4,8,16,52}` grid
   and more origins (compute allowing, possibly outside this sandbox) would give a
   real-confidence answer instead of a directional one.
3. **Investigate the profile-metric mechanism directly**: section 3's aggregation-
   cancels-noise explanation is plausible but not verified here — a targeted check
   (e.g. profile error vs. session count per cell) could confirm or refute it.
4. **Resolve the `all_results.parquet` provenance gap** noted above before it's
   cited again as ground truth.
5. **If pursued further, formal harness integration is a separate, later session**
   (mirroring how Session 3's ad-hoc VAE arm became Session 4's formal `benchmark.py`
   wiring) — not something to fold into this diagnostic script.

## Reproducing this investigation

```bash
python scripts/validate_shared_vae_hypothesis.py \
  --data data/preprocessed_data/office.csv --dataset-name office \
  --x-grid 1,4,16 --horizons 1,2,3 --n-origins 5 --step-days 7 --n-scenarios 15 \
  --vae-epochs 50 --vae-hidden-dim 256 --vae-latent-dim 16 \
  --out results/persistence_gap/office_shared_vae.csv

python scripts/validate_shared_vae_hypothesis.py \
  --data data/preprocessed_data/sap.csv --dataset-name sap \
  --x-grid 4,16 --horizons 1,2 --n-origins 2 --step-days 7 --n-scenarios 10 \
  --vae-epochs 40 --vae-hidden-dim 256 --vae-latent-dim 16 \
  --out results/persistence_gap/sap_shared_vae.csv

python scripts/validate_shared_vae_hypothesis.py \
  --data data/preprocessed_data/boulder.csv --dataset-name boulder \
  --x-grid 4,16 --horizons 1,2 --n-origins 2 --step-days 7 --n-scenarios 10 \
  --vae-epochs 40 --vae-hidden-dim 256 --vae-latent-dim 16 \
  --out results/persistence_gap/boulder_shared_vae.csv

python scripts/validate_shared_vae_hypothesis.py \
  --data data/sample_df.pkl --dataset-name sample_df_home_dept92 \
  --filter "location_type=='home' and department=='92'" \
  --x-grid 16 --horizons 1,2 --n-origins 1 --step-days 7 --n-scenarios 10 \
  --vae-epochs 40 --vae-hidden-dim 256 --vae-latent-dim 16 \
  --out results/persistence_gap/sample_df_shared_vae.csv
```

Combined output: `results/persistence_gap/combined_all_datasets.csv` (1,051 rows,
concatenation of all four runs above).
