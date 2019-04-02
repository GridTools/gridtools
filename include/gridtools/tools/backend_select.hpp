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

#include "../common/defs.hpp"

#if GT_FLOAT_PRECISION == 4
using float_type = float;
#elif GT_FLOAT_PRECISION == 8
using float_type = double;
#else
#error float precision not properly set (4 or 8 bytes supported)
#endif

#ifdef GT_BACKEND_X86
using backend_t = gridtools::backend::x86;
#elif defined(GT_BACKEND_NAIVE)
using backend_t = gridtools::backend::naive;
#elif defined(GT_BACKEND_MC)
using backend_t = gridtools::backend::mc;
#elif defined(GT_BACKEND_CUDA)
using backend_t = gridtools::backend::cuda;
#endif
