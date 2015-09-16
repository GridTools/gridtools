#pragma once

#ifdef RECTANGULAR_GRIDS
    #include "stencil-composition/rectangular_grids/accessor_fwd.hpp"
#else
    #include "stencil-composition/other_grids/accessor_fwd.hpp"
#endif

namespace gridtools {
    template<typename T>
    struct is_accessor;

    /**
     * Struct to test if an argument is a temporary
     */
    template <typename T>
    struct is_plchldr_to_temp;

}
