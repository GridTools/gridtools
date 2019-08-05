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
#include <type_traits>

#ifndef GT_DEFAULT_VERTICAL_BLOCK_SIZE
#define GT_DEFAULT_VERTICAL_BLOCK_SIZE 20
#endif

namespace gridtools {
    namespace execute {
        template <uint_t BlockSize>
        struct parallel_block {
            static constexpr uint_t block_size = BlockSize;
        };
        using parallel = parallel_block<GT_DEFAULT_VERTICAL_BLOCK_SIZE>;
        struct forward {};
        struct backward {};

        template <typename T>
        struct is_parallel : std::false_type {};

        template <uint_t BlockSize>
        struct is_parallel<parallel_block<BlockSize>> : std::true_type {};

        template <typename T>
        struct is_forward : std::false_type {};

        template <>
        struct is_forward<forward> : std::true_type {};

        template <typename T>
        struct is_backward : std::false_type {};

        template <>
        struct is_backward<backward> : std::true_type {};
    } // namespace execute

    template <typename T>
    struct is_execution_engine : std::false_type {};

    template <uint_t BlockSize>
    struct is_execution_engine<execute::parallel_block<BlockSize>> : std::true_type {};

    template <>
    struct is_execution_engine<execute::forward> : std::true_type {};

    template <>
    struct is_execution_engine<execute::backward> : std::true_type {};

} // namespace gridtools
