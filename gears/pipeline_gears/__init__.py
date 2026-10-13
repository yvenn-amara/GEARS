"""Per-GEAR backends dispatched to by :class:`gears.pipeline.GEARSModel`.

Only GEAR 1st (:class:`~gears.pipeline_gears.gear1.Gear1Backend`) is implemented today.
"""

from gears.pipeline_gears.gear1 import Gear1Backend

__all__ = ["Gear1Backend"]
