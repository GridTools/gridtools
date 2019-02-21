/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <gtest/gtest.h>

extern "C" void call_repository(); // implemented in test_repository.f90
TEST(repository_with_custom_getter_prefix, fortran_bindings) {
    // the test for this code is in exported_repository.cpp
    call_repository();
}
