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
#include "../stencil-composition/backend.hpp"

#if GT_FLOAT_PRECISION == 4
using float_type = float;
#elif GT_FLOAT_PRECISION == 8
using float_type = double;
#else
#error float precision not properly set (4 or 8 bytes supported)
#endif

#ifdef GT_BACKEND_X86
using target_t = gridtools::target::x86;
#ifdef GT_BACKEND_STRATEGY_NAIVE
using strategy_t = gridtools::strategy::naive;
#else
using strategy_t = gridtools::strategy::block;
#endif
#elif defined(GT_BACKEND_MC)
using target_t = gridtools::target::mc;
using strategy_t = gridtools::strategy::block;
#elif defined(GT_BACKEND_CUDA)
using target_t = gridtools::target::cuda;
using strategy_t = gridtools::strategy::block;
#else
#define GT_NO_BACKEND
#endif

#ifndef GT_NO_BACKEND
using backend_t = gridtools::backend<target_t>;
#endif
