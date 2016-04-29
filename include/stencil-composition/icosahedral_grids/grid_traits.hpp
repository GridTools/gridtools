#pragma once
#include <boost/mpl/quote.hpp>

#include "compute_extents_metafunctions.hpp"
#include "icosahedral_grid_traits.hpp"

namespace gridtools {

    template <>
    struct grid_traits_from_id< enumtype::icosahedral > {
        struct select_mss_compute_extent_sizes {
            typedef boost::mpl::quote1< icgrid::mss_compute_extent_sizes > type;
        };

        typedef extent< 0 > null_extent_t;

        template < enumtype::platform BackendId >
        struct with_arch {
            typedef icgrid::grid_traits_arch< BackendId > type;
        };

        typedef static_uint< 0 > dim_i_t;
        typedef static_uint< 1 > dim_c_t;
        typedef static_uint< 2 > dim_j_t;
        typedef static_uint< 3 > dim_k_t;
    };
}
