/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
/**
   @file
   @brief File containing the definition of the placeholders used to address the storage from whithin the functors.
   A placeholder is an implementation of the proxy design pattern for the storage class, i.e. it is a light object used
   in place of the storage when defining the high level computations,
   and it will be bound later on with a specific instantiation of a storage class.
*/

#pragma once

#include <cstddef>
#include <type_traits>

#include "../common/integral_constant.hpp"
#include "../meta.hpp"

namespace gridtools {
    template <size_t, class Data>
    struct tmp_arg {
        using data_t = Data;
        using num_colors_t = integral_constant<int_t, 1>;
        using tmp_tag = std::true_type;
    };

    template <class T, class = void>
    struct is_tmp_arg : std::false_type {};

    template <class T>
    struct is_tmp_arg<T, void_t<typename T::tmp_tag>> : std::true_type {};
} // namespace gridtools
