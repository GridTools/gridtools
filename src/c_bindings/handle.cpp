/*
 * GridTools
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/c_bindings/handle.h>
#include <gridtools/c_bindings/handle_impl.hpp>

void gt_release(gt_handle const *obj) { delete obj; }
