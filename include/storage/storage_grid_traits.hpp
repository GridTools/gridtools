#pragma once

namespace gridtools {

#ifdef STRUCTURED_GRIDS
    struct storage_grid_traits {

        typedef static_uint< 0 > dim_i_t;
        typedef static_uint< 1 > dim_j_t;
        typedef static_uint< 2 > dim_k_t;
    };
#else
    struct storage_grid_traits {

        typedef static_uint< 0 > dim_i_t;
        typedef static_uint< 1 > dim_c_t;
        typedef static_uint< 2 > dim_j_t;
        typedef static_uint< 3 > dim_k_t;
    };
#endif
}
