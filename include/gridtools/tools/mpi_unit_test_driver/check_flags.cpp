/*
 * GridTools
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once
#ifdef __CUDACC__
#ifndef GCL_GPU
#define GCL_GPU
#endif
#else
#ifdef GCL_GPU
#undef GCL_GPU
#endif
#endif
