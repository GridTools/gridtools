#pragma once 

#ifdef RECTANGULAR_GRIDS
    #ifdef CXX11_ENABLED
        #include "stencil-composition/rectangular_grids/make_esf_cxx11.hpp"
    #else
        #include "stencil-composition/rectangular_grids/make_esf_cxx03.hpp"
    #endif
#else
        #include "stencil-composition/other_grids/make_esf.hpp"
#endif
