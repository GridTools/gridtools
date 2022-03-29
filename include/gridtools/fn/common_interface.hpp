/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include "./backend2/common.hpp"

namespace gridtools::fn {

    template <class T, class Allocator, class Sizes>
    auto allocate_global_tmp(Allocator &alloc, Sizes const &sizes) {
        return allocate_global_tmp(alloc, sizes, backend::data_type<T>());
    }

} // namespace gridtools::fn
