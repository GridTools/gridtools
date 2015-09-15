#pragma once

#ifdef RECTANGULAR_GRIDS
    #include "stencil-composition/rectangular_grids/backend_metafunctions.hpp"
#else
    #include "../experimental/grids/backend_metafunctions.hpp"
#endif
