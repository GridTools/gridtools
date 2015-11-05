#pragma once

#include "stencil-composition/compute_ranges_metafunctions.hpp"

namespace gridtools {
#ifdef RECTANGULAR_GRIDS
    struct select_mss_compute_range_sizes
    {
        typedef boost::mpl::quote1<recgrid::mss_compute_range_sizes> type;
    };
#else
    struct select_mss_compute_range_sizes
    {
        typedef boost::mpl::quote1<othergrid::mss_compute_range_sizes> type;
    };
#endif

} // namespace gridtools
