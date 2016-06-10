#pragma once

#ifdef STRUCTURED_GRIDS
#include "stencil_composition/structured_grids/backend_cuda/execute_kernel_functor_cuda.hpp"
#else
#include "stencil_composition/icosahedral_grids/backend_cuda/execute_kernel_functor_cuda.hpp"
#endif
