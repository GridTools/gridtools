#pragma once

#ifdef STRUCTURED_GRIDS
#include "stencil-composition/structured_grids/backend_host/execute_kernel_functor_host.hpp"
#else
#include "stencil-composition/icosahedral_grids/backend_host/execute_kernel_functor_host.hpp"
#endif
