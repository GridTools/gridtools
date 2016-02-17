#pragma once

#include <boost/mpl/quote.hpp>
#include "compute_extents_metafunctions.hpp"
#include "grid_traits_backend_fwd.hpp"

#ifdef __CUDACC__
#include "backend_cuda/grid_traits_cuda.hpp"
#else
#include "backend_host/grid_traits_host.hpp"
#endif

namespace gridtools {

    template <>
    struct grid_traits_from_id<enumtype::structured> {

        struct select_mss_compute_extent_sizes {
            typedef boost::mpl::quote1< strgrid::mss_compute_extent_sizes > type;
        };

        typedef extent<0,0,0,0> null_extent_t;

        template<enumtype::platform BackendId>
        struct with_arch
        {
            typedef strgrid::grid_traits_arch<BackendId> type;
        };

    };

}
