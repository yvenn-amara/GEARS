"""Per-GEAR backends dispatched to by :class:`gears.pipeline.GEARSModel`.

Only GEAR 1st (:class:`~gears.pipeline_gears.gear1.Gear1Backend`) is implemented today; GEAR 2nd to 5th
are reserved. A new gear is a backend class in this package, registered in ``gears.pipeline._GEAR_BACKENDS``
(see CONTRIBUTING.md, "Adding a gear").
"""

from gears.pipeline_gears.gear1 import Gear1Backend

__all__ = ["Gear1Backend"]
