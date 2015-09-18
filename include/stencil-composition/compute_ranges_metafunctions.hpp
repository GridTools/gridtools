#pragma once

#ifdef RECTANGULAR_GRIDS
    #include "stencil-composition/rectangular_grids/compute_ranges_metafunctions.hpp"
#else
    #include "stencil-composition/other_grids/compute_ranges_metafunctions.hpp"
#endif
