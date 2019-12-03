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

// stencil composition backend
#ifdef GT_BACKEND_X86
#include "../stencil_composition/backend/x86.hpp"
using backend_t = gridtools::x86::backend<>;
#elif defined(GT_BACKEND_NAIVE)
#include "../stencil_composition/backend/naive.hpp"
using backend_t = gridtools::naive::backend;
#elif defined(GT_BACKEND_MC)
#ifndef GT_ICOSAHEDRAL_GRIDS
#include "../stencil_composition/backend/mc.hpp"
using backend_t = gridtools::mc::backend;
#endif
#elif defined(GT_BACKEND_CUDA)
#ifdef __CUDACC__
#include "../stencil_composition/backend/cuda.hpp"
using backend_t = gridtools::cuda::backend<>;
#endif
#endif

// default timer implementation
#ifdef GT_BACKEND_CUDA
#include "../common/timer/timer_cuda.hpp"
using timer_impl_t = gridtools::timer_cuda;
#else
#include "../common/timer/timer_omp.hpp"
using timer_impl_t = gridtools::timer_omp;
#endif

// gcl arch
#include "../communication/low_level/gcl_arch.hpp"
#ifdef GT_BACKEND_CUDA
using gcl_arch_t = gridtools::gcl_gpu;
#else
using gcl_arch_t = gridtools::gcl_cpu;
#endif

// storage
#if defined(GT_BACKEND_CUDA)
#include "../storage/cuda.hpp"
using storage_traits_t = gridtools::storage::cuda;
#elif defined(GT_BACKEND_X86) || defined(GT_BACKEND_NAIVE)
#include "../storage/x86.hpp"
using storage_traits_t = gridtools::storage::x86;
#elif defined(GT_BACKEND_MC)
#include "../storage/mc.hpp"
using storage_traits_t = gridtools::storage::mc;
#endif
