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

#ifdef __CUDACC__
#include "backend_cuda/grid_traits_cuda.hpp"
#endif
#include "backend_x86/grid_traits_x86.hpp"
