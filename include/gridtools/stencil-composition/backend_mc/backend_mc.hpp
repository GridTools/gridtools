/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#ifndef GT_ICOSAHEDRAL_GRIDS

// This file contains all header files required by the mc backend
#ifdef _OPENMP
#include <omp.h>
#endif

#include "../../storage/storage-facility.hpp"
#include "../backend_fwd.hpp"
#include "../iteration_policy.hpp"
#include "../structured_grids/backend_mc/strategy_mc.hpp"
#include "./backend_traits_mc.hpp"

#endif
