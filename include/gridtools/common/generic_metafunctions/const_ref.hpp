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

#include "../../meta/macros.hpp"
#include "../../meta/type_traits.hpp"

namespace gridtools {
    GT_META_LAZY_NAMESPACE {
        template <class T, class = void>
        struct const_ref {
            using type = T const &;
        };
        template <class T>
        struct const_ref<T,
            enable_if_t<std::is_trivially_copy_constructible<decay_t<T>>::value &&
                        sizeof(decay_t<T>) <= sizeof(add_pointer_t<T>)>> {
            using type = decay_t<T>;
        };
    }
    GT_META_DELEGATE_TO_LAZY(const_ref, class T, T);
} // namespace gridtools
