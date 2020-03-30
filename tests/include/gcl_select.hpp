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

#include <gridtools/communication/low_level/gcl_arch.hpp>

#if defined(GT_GCL_GPU)
using gcl_arch_t = gridtools::gcl_gpu;
#elif defined(GT_GCL_CPU)
using gcl_arch_t = gridtools::gcl_cpu;
#endif
