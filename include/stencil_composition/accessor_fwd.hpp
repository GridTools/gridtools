#pragma once

#ifdef STRUCTURED_GRIDS
#include "stencil_composition/structured_grids/accessor_fwd.hpp"
#else
#include "stencil_composition/icosahedral_grids/accessor_fwd.hpp"
#endif

namespace gridtools {
    template < typename T >
    struct is_accessor;

    /**
     * Struct to test if an argument is a temporary
     */
    template < typename T >
    struct is_plchldr_to_temp;
}
