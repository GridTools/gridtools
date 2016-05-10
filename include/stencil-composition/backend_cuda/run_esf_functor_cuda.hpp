#pragma once

#ifdef STRUCTURED_GRIDS
#include "../structured_grids/backend_cuda/run_esf_functor_cuda.hpp"
#else
#include "../icosahedral_grids/backend_cuda/run_esf_functor_cuda.hpp"
#endif
