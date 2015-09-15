#pragma once

#ifdef RECTANGULAR_GRIDS
    #include "stencil-composition/rectangular_grids/backend.hpp"
#else
    #include "../experimental/grids/backend.hpp"
#endif
