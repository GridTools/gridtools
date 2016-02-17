#pragma once

#ifdef __CUDACC__
#include "backend_cuda/icosahedral_grid_traits_cuda.hpp"
#else
#include "backend_host/icosahedral_grid_traits_host.hpp"
#endif
