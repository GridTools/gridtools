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

#include <type_traits>

#include "../common/defs.hpp"
#include "../common/host_device.hpp"
#include "../common/integral_constant.hpp"

#ifndef GT_DEFAULT_VERTICAL_BLOCK_SIZE
#define GT_DEFAULT_VERTICAL_BLOCK_SIZE 1
#endif

namespace gridtools {
    namespace execute {
        template <int_t BlockSize>
        struct parallel_block {
            static constexpr int_t block_size = BlockSize;
        };
        using parallel = parallel_block<GT_DEFAULT_VERTICAL_BLOCK_SIZE>;
        struct forward {};
        struct backward {};

        template <typename T>
        struct is_parallel : std::false_type {};

        template <int_t BlockSize>
        struct is_parallel<parallel_block<BlockSize>> : std::true_type {};

        template <typename T>
        struct is_forward : std::false_type {};

        template <>
        struct is_forward<forward> : std::true_type {};

        template <typename T>
        struct is_backward : std::false_type {};

        template <>
        struct is_backward<backward> : std::true_type {};

        template <typename T>
        constexpr integral_constant<int_t, is_backward<T>::value ? -1 : 1> step = {};

        template <typename T>
        struct block_size : integral_constant<int_t, 0> {};

        template <int_t BlockSize>
        struct block_size<parallel_block<BlockSize>> : integral_constant<int_t, BlockSize> {};
    } // namespace execute

    template <typename T>
    struct is_execution_engine : std::false_type {};

    template <int_t BlockSize>
    struct is_execution_engine<execute::parallel_block<BlockSize>> : std::true_type {};

    template <>
    struct is_execution_engine<execute::forward> : std::true_type {};

    template <>
    struct is_execution_engine<execute::backward> : std::true_type {};

} // namespace gridtools
