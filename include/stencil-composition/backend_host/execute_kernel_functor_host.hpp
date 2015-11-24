#pragma once

#ifdef RECTANGULAR_GRIDS
    #include "stencil-composition/rectangular_grids/backend_host/execute_kernel_functor_host.hpp"
#else
    #include "stencil-composition/other_grids/backend_host/execute_kernel_functor_host.hpp"
#endif
