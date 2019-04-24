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
#ifdef __CUDACC__
#include "structured_grids/backend_cuda/mss_loop_cuda.hpp"
#endif
#include "structured_grids/backend_mc/mss_loop_mc.hpp"
#include "structured_grids/backend_x86/mss_loop_x86.hpp"
#else
#ifdef __CUDACC__
#include "icosahedral_grids/backend_cuda/mss_loop_cuda.hpp"
#endif
#include "icosahedral_grids/backend_x86/mss_loop_x86.hpp"
#endif
