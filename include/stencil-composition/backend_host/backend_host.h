#pragma once

// This file contains all header files required by the host backend
#ifdef _OPENMP
    #include <omp.h>
#endif

#include "../execution_policy.h"
#include "../heap_allocated_temps.h"
#include "../iteration_policy.h"
#include "../backend_fwd.h"
#include "backend_traits_host.h"
#include "../loop_hierarchy.h"
#include "strategy_host.h"


