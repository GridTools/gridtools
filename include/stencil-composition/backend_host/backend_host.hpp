/*
   Copyright 2016 GridTools Consortium

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
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
