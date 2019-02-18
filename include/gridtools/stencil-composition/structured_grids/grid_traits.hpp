/*
 * GridTools Libraries
 * Copyright (c) 2019, ETH Zurich
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#ifdef __CUDACC__
#include "./backend_cuda/grid_traits_cuda.hpp"
#endif
#include "./backend_mc/grid_traits_mc.hpp"
#include "./backend_x86/grid_traits_x86.hpp"
