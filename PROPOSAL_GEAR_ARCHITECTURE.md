# GEAR-Level Architecture Proposal — Phase 2 / Session 1

**Status:** proposal for review — no dispatch mechanism implemented yet.
**GEAR 2nd sanity check (answered 2026-08-02):** its primary `fit` entry point will still take a
pandas DataFrame (or a path to one) as its main input, even though everything after that differs
from GEAR 1st. This proposal is designed around that answer — it standardizes only "the first
thing you fit on is data," and deliberately does not standardize anything else.

## The finding this proposal starts from

Today, `GEARSModel` *is* what this proposal calls "GEAR 1st" — there's no `gear=` concept at all;
`gears/pipeline.py`'s `GEARSModel.__init__` and `.fit()` directly implement the full
GMM/VAE-session-modeling → SARIMA/probabilistic-forecasting → simulation → smart-charging
pipeline. There is no seam anywhere for a structurally different model family to plug in.

A second, related finding (confirmed by reading `gears/pipeline.py`): **`GEARSModel.fit()` does
not expose `model_type` or `recency` at all.** VAE and recency-weighted GMM are currently only
reachable by constructing `EVSessionGMM`/`EVSessionModel` directly and bypassing the "unified
facade" entirely — so today's `GEARSModel` isn't even a complete facade over GEAR 1st's own
capabilities. This proposal folds the fix into the same design, rather than treating it as
unrelated cleanup.

## Design

### 1. `GEARSModel` becomes a thin, gear-dispatching facade

```python
class GEARSModel:
    def __init__(self, gear: int = 1, **kwargs):
        if gear not in _GEAR_BACKENDS:
            raise NotImplementedError(
                f"GEAR {gear} is reserved for a future release and is not implemented yet. "
                f"Only GEAR 1 is available today. Implemented gears: "
                f"{sorted(_GEAR_BACKENDS)}."
            )
        self.gear = gear
        self._backend = _GEAR_BACKENDS[gear](**kwargs)

    def fit(self, data, *args, **kwargs):
        return self._backend.fit(data, *args, **kwargs)

    def simulate_short_term(self, *args, **kwargs):
        return self._backend.simulate_short_term(*args, **kwargs)

    def simulate_medium_term(self, *args, **kwargs):
        return self._backend.simulate_medium_term(*args, **kwargs)

    def export(self, *args, **kwargs):
        return self._backend.export(*args, **kwargs)

    # ... other current GEARSModel public methods, same pattern
```

`_GEAR_BACKENDS = {1: Gear1Backend}` today; gears 2–5 simply aren't in the dict, which is what
produces the friendly `NotImplementedError` in `__init__` — no separate `if gear == 2: raise ...`
branches to maintain per gear.

### 2. Why this doesn't lock future gears into today's signatures

Beyond `fit`'s first positional argument (`data`), **every dispatcher method forwards `*args,
**kwargs` untouched** — it does not declare or validate parameter names for
`simulate_short_term`/`simulate_medium_term`/`export`/anything else. A future `Gear2Backend` is
free to define `simulate_short_term(self, horizon_days, *, confidence_level=0.9)` or something
with no resemblance to GEAR 1st's `simulate_short_term(start_date, horizon)` at all — the facade
doesn't know or care, it just forwards. There is deliberately **no `abc.ABC` / `typing.Protocol`
base class** fixing an abstract interface across gears, because that would re-introduce exactly
the constraint the user has already said not to guess at (GEAR 2nd's internals are unknown, and
its `fit`/`simulate_*` semantics were flagged as likely to differ).

The one exception is `fit`'s first argument: since GEAR 2nd is confirmed to still take a
DataFrame-or-path as its primary input, the facade standardizes on `fit(self, data, ...)` as a
named first parameter (rather than swallowing it into `*args`) purely so that `GEARSModel(gear=1,
...).fit(df)` and a future `GEARSModel(gear=2, ...).fit(df)` read the same way at the call site —
this is a documentation/consistency convenience, not an enforced contract, and it costs nothing
to relax later if a future gear turns out not to need it.

### 3. `Gear1Backend` — today's pipeline, unchanged behavior, moved behind the seam

`Gear1Backend` is `gears/pipeline.py`'s current `GEARSModel` class body, moved essentially as-is
into a new `gears/pipeline_gears/gear1.py` (or similar — exact module layout is an implementation
detail for Session 2, not this proposal). `GEARSModel(gear=1, ...)` should be behavior-identical
to today's `GEARSModel(...)` for every existing caller — this is a refactor of the seam, not a
behavior change.

**Folded-in fix**: `Gear1Backend.__init__` gains `model_type: Literal["gmm", "vae"] = "gmm"`,
`recency: bool = False`, and `half_life_days: float | None = None` as first-class constructor
parameters (matching `EVSessionModel`'s own parameter names exactly, so there's one obvious place
each lives). `Gear1Backend.fit()` passes them through when constructing the underlying
`EVSessionModel`. This closes the gap where VAE/recency were only reachable by bypassing the
facade — `GEARSModel(gear=1, model_type="vae").fit(df)` becomes a supported, documented path.

### 4. Gears 2–5

Not designed here, on purpose — the user has confirmed GEAR 2nd's model family is structurally
different and its details aren't yet available. All this session commits to is: (a) the
constructor raises a clear `NotImplementedError` naming which gears exist today, or an
authoritative-object with roadmap link; and (b) the day GEAR 2nd is ready, its
`fit(data, ...)`-shaped backend can be added by writing a new `GearNBackend` class and one dict
entry — no changes needed to `GEARSModel` itself.

## Acceptance / open items for review

- Confirm `_GEAR_BACKENDS` as a plain dict (not an entry-point/plugin system) is the right amount
  of machinery for now — a plugin system would be over-engineering for a package with one
  implemented gear.
- Confirm folding `model_type`/`recency`/`half_life_days` into `Gear1Backend.__init__` (rather
  than `.fit()`) is the right home for them — constructor placement means they're fixed for the
  lifetime of one `GEARSModel` instance, matching how `EVSessionModel` itself treats them today.
- Module layout for where `Gear1Backend`'s code physically lives is left to Session 2's judgment;
  this proposal only fixes the *shape* of the dispatch, not the file tree.
