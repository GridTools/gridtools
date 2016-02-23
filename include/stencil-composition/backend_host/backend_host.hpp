#pragma once

// This file contains all header files required by the host backend
#ifdef _OPENMP
    #include <omp.h>
#endif

#include "stencil-composition/execution_policy.hpp"
#include "stencil-composition/heap_allocated_temps.hpp"
#include "stencil-composition/iteration_policy.hpp"
#include "stencil-composition/backend_fwd.hpp"
#include "stencil-composition/backend_host/backend_traits_host.hpp"
#include "stencil-composition/loop_hierarchy.hpp"
#include "stencil-composition/backend_host/strategy_host.hpp"
