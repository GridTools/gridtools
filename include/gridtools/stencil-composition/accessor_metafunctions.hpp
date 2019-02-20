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

#include <type_traits>

#include "../common/defs.hpp"
#include "../meta/type_traits.hpp"
#include "accessor_intent.hpp"
#include "is_accessor.hpp"

namespace gridtools {
    template <class Accessor, class = void>
    struct is_accessor_readonly : std::false_type {};

    template <class Accessor>
    struct is_accessor_readonly<Accessor, enable_if_t<is_accessor<Accessor>::value>>
        : bool_constant<Accessor::intent_v == intent::in> {};

    template <class Accessor, class = void>
    struct is_accessor_written : std::false_type {};

    template <class Accessor>
    struct is_accessor_written<Accessor, enable_if_t<is_accessor<Accessor>::value>>
        : bool_constant<Accessor::intent_v == intent::inout> {};

    template <class Accessor>
    struct accessor_index {
        GT_STATIC_ASSERT((is_accessor<Accessor>::value), GT_INTERNAL_ERROR);
        using type = typename Accessor::index_t;
    };
} // namespace gridtools
