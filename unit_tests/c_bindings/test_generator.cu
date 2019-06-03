/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

// Unittest disabled for the combination nvcc + clang with nvcc < 10.0 because of problems with operator""if
#if not(defined __clang__ && __CUDACC_VER_MAJOR < 10)
#include "test_generator.cpp"
#endif
