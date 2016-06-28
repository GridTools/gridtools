#pragma once

#ifdef STRUCTURED_GRIDS
#include "stencil-composition/structured_grids/backend_cuda/execute_kernel_functor_cuda.hpp"
#else
#include "stencil-composition/icosahedral_grids/backend_cuda/execute_kernel_functor_cuda.hpp"
#endif
