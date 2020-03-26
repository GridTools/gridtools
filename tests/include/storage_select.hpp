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

namespace gridtools {
    namespace storage {
        struct cuda;
        struct x86;
        struct mc;
    } // namespace storage
} // namespace gridtools

#if defined(GT_STORAGE_CUDA)
#include <gridtools/storage/cuda.hpp>
namespace {
    using storage_traits_t = gridtools::storage::cuda;
}
#elif defined(GT_STORAGE_X86)
#include <gridtools/storage/x86.hpp>
namespace {
    using storage_traits_t = gridtools::storage::x86;
}
#elif defined(GT_STORAGE_MC)
#include <gridtools/storage/mc.hpp>
namespace {
    using storage_traits_t = gridtools::storage::mc;
}
#endif
