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

// This file contains all header files required by the host backend
#ifdef _OPENMP
#include <omp.h>
#endif

#include "../../storage/storage_facility.hpp"
#include "../backend_fwd.hpp"
#include "../iteration_policy.hpp"
#include "backend_traits_x86.hpp"
