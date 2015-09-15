#pragma once 

#ifdef RECTANGULAR_GRIDS
    #include "stencil-composition/rectangular_grids/make_esf.hpp"
#else
        #include "../../experimental/grids/make_esf.hpp"
#endif
