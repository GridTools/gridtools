#pragma once 

#ifdef RECTANGULAR_GRIDS
    #include "stencil-composition/rectangular_grids/make_esf.hpp"
#else
        #include "stencil-composition/other_grids/make_esf.hpp"
#endif
