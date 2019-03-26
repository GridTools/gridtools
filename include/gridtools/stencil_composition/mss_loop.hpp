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
#include "./backend_cuda/mss_loop_cuda.hpp"
#endif
#ifndef GT_ICOSHEDRAL_GRIDS
#include "./backend_mc/mss_loop_mc.hpp"
#endif
#include "./backend_naive/mss_loop_naive.hpp"
#include "./backend_x86/mss_loop_x86.hpp"
