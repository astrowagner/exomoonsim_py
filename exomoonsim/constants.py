"""Physical constants and unit conversions (mirrors the values in exomoonsim.pro)."""

# length
AU2M   = 149_597_870_700.0   # au -> m
RSUN2M = 695_700_000.0        # solar radii -> m
RJUP2M = 71_492_000.0         # Jupiter radii -> m
REAR2M = 6_378_000.0          # Earth radii -> m

# mass
MSUN2KG = 1.988e30
MJUP2KG = 1.898e27
MEAR2KG = 5.972e24

# handy derived
RJUP_PER_AU = AU2M / RJUP2M   # Jupiter radii in one au
