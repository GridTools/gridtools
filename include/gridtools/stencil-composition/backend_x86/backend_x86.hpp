/*
 * GridTools Libraries
 * Copyright (c) 2019, ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

// This file contains all header files required by the host backend
#ifdef _OPENMP
#include <omp.h>
#endif

#include "../../storage/storage-facility.hpp"
#include "../backend_fwd.hpp"
#include "../iteration_policy.hpp"
#include "backend_traits_x86.hpp"
#include "strategy_x86.hpp"
