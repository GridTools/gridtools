#pragma once
#include <gridtools.hpp>

namespace vertical_advection {

// define some physical constants
#define BETA_V ((double)0.0)
#define BET_M ((double)0.5 * ((double)1.0 - BETA_V))
#define BET_P ((double)0.5 * ((double)1.0 + BETA_V))

#ifdef CUDA_EXAMPLE
    typedef gridtools::backend< gridtools::enumtype::Cuda, gridtools::GRIDBACKEND, gridtools::enumtype::Block >
        va_backend;
#else
#ifdef BACKEND_BLOCK
    typedef gridtools::backend< gridtools::enumtype::Host, gridtools::GRIDBACKEND, gridtools::enumtype::Block >
        va_backend;
#else
    typedef gridtools::backend< gridtools::enumtype::Host, gridtools::GRIDBACKEND, gridtools::enumtype::Naive >
        va_backend;
#endif
#endif
}
