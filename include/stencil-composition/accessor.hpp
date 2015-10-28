#pragma once

#ifdef RECTANGULAR_GRIDS
    #include "stencil-composition/rectangular_grids/accessor_metafunctions.hpp"
#include "stencil-composition/rectangular_grids/accessor.hpp"
#else
    #include "stencil-composition/other_grids/accessor_metafunctions.hpp"
    #include "stencil-composition/other_grids/accessor.hpp"
#endif
