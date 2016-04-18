#pragma once

#include "grid_traits_fwd.hpp"

#ifdef STRUCTURED_GRIDS
#include "structured_grids/grid_traits.hpp"
#else
#include "icosahedral_grids/grid_traits.hpp"
#endif
