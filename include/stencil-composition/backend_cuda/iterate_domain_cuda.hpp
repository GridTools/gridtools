#pragma once

#ifdef STRUCTURED_GRIDS
    #include "../structured_grids/backend_cuda/iterate_domain_cuda.hpp"
#else
    #include "../icosahedral_grids/backend_cuda/iterate_domain_cuda.hpp"
#endif
