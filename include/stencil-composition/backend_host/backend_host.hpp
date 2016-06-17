#pragma once

// This file contains all header files required by the host backend
#ifdef _OPENMP
#include <omp.h>
#endif

#include "../execution_policy.hpp"
#include "../heap_allocated_temps.hpp"
#include "../iteration_policy.hpp"
#include "../backend_fwd.hpp"
#include "../../storage/storage-facility.hpp"
#include "backend_traits_host.hpp"
#include "../loop_hierarchy.hpp"
#include "strategy_host.hpp"
