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

// stencil composition backend
#ifdef GT_BACKEND_X86
#include <gridtools/stencil_composition/backend/x86.hpp>
using backend_t = gridtools::x86::backend<>;
#elif defined(GT_BACKEND_NAIVE)
#include <gridtools/stencil_composition/backend/naive.hpp>
using backend_t = gridtools::naive::backend;
#elif defined(GT_BACKEND_MC)
#include <gridtools/stencil_composition/backend/mc.hpp>
using backend_t = gridtools::mc::backend;
#elif defined(GT_BACKEND_CUDA)
#include <gridtools/stencil_composition/backend/cuda.hpp>
using backend_t = gridtools::cuda::backend<>;
#endif
