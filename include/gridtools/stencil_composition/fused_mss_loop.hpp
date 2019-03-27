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
#include "./backend_cuda/fused_mss_loop_cuda.hpp"
#endif
#ifndef GT_ICOSAHEDRAL_GRIDS
#include "./backend_mc/fused_mss_loop_mc.hpp"
#endif
#include "./backend_naive/fused_mss_loop_naive.hpp"
#include "./backend_x86/fused_mss_loop_x86.hpp"
