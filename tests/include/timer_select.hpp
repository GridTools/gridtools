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

// default timer implementation
#if defined(GT_TIMER_CUDA)
#include <gridtools/common/timer/timer_cuda.hpp>
using timer_impl_t = gridtools::timer_cuda;
#elif defined(GT_TIMER_OMP)
#include <gridtools/common/timer/timer_omp.hpp>
using timer_impl_t = gridtools::timer_omp;
#elif defined(GT_TIMER_DUMMY)
#include <gridtools/common/timer/timer_dummy.hpp>
using timer_impl_t = gridtools::timer_dummy;
#endif
