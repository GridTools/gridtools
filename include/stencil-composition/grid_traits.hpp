#pragma once

#include "stencil-composition/compute_extents_metafunctions.hpp"

namespace gridtools {
#ifdef STRUCTURED_GRIDS
    struct select_mss_compute_extent_sizes
    {
        typedef boost::mpl::quote1<strgrid::mss_compute_extent_sizes> type;
    };
#else
    struct select_mss_compute_extent_sizes
    {
        typedef boost::mpl::quote1<icgrid::mss_compute_extent_sizes> type;
    };
#endif

} // namespace gridtools
