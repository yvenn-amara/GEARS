"""
Per-GEAR backend implementations dispatched to by :class:`gears.pipeline.GEARSModel`.

See ``PROPOSAL_GEAR_ARCHITECTURE.md`` at the repo root for the design this
package implements. Today only GEAR 1st (:class:`~gears.pipeline_gears.gear1.Gear1Backend`)
is implemented; GEAR 2nd-5th are reserved for future, structurally different
model families.
"""

from gears.pipeline_gears.gear1 import Gear1Backend

__all__ = ["Gear1Backend"]
