#pragma once

#include "stencil-composition/compute_ranges_metafunctions.hpp"

namespace gridtools {
#ifdef STRUCTURED_GRIDS
    struct select_mss_compute_range_sizes
    {
        typedef boost::mpl::quote1<structgrid::mss_compute_range_sizes> type;
    };
#else
    struct select_mss_compute_range_sizes
    {
        typedef boost::mpl::quote1<icosgrid::mss_compute_range_sizes> type;
    };
#endif

} // namespace gridtools
