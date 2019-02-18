/*
 * GridTools Libraries
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#ifdef GT_STRUCTURED_GRIDS
#include "../structured_grids/backend_cuda/execute_kernel_functor_cuda.hpp"
#else
#include "../icosahedral_grids/backend_cuda/execute_kernel_functor_cuda.hpp"
#endif
